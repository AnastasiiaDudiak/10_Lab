"""Спільні утиліти ЛР 10: шляхи, I/O, токенізація для BM25, обгортка embeddings, кеш."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
DOCS_DIR = ROOT / "docs"
EVAL = ROOT / "eval"
for _p in (DATA, RAW, CACHE, DOCS_DIR, EVAL):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# Моделі embeddings (обидві багатомовні, підтримують українську).
MODELS = {
    "e5-small": {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",      # e5 вимагає префікси (див. model card)
        "passage_prefix": "passage: ",
    },
    "minilm": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
}
DEFAULT_MODEL = "e5-small"
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"   # багатомовний cross-encoder

# Параметри HNSW
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------- I/O
def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_text(path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    print(f"[write] {Path(path).relative_to(ROOT)}")


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in r) + " |")
    return "\n".join(out)


def load_gold(path=None) -> list[dict]:
    path = Path(path or EVAL / "gold.jsonl")
    if not path.exists():
        raise SystemExit(f"Немає {path}. Спершу потрібен золотий набір (eval/gold.jsonl).")
    return read_jsonl(path)


def gold_subset(gold, name):
    """Підмножина запитів (chunking=10, embeddings=20), лише з відповіддю."""
    return [g for g in gold if name in g.get("subsets", []) and g.get("relevant")]


# ---------------------------------------------------------------- BM25 tokens
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ'’]+")
STOPWORDS = set("""
і й та а але або чи що як це той ця ці цей то так також у в на з із зі до від по за для про при під над між через
не ні же б би бо є був була було були буде мають має який яка яке які якого якої яких де коли там тут його її їх
вони він вона воно ми ви я ти свій своя своє свої все всі весь вся щоб ще вже лише тільки саме ж
the a an and or of to in on for by with is are was were be been as at from that this these those it its into
""".split())


def bm25_tokens(text: str, stem: bool = True) -> list[str]:
    """Токенізація для BM25: нижній регістр, стоп-слова, «легкий стемінг» — обрізання до 6 символів
    (компенсує відмінки української: «землетрусу», «землетрусом» -> «землет»)."""
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        t = t.strip("'’")
        if not t or t in STOPWORDS:
            continue
        if stem and not t.isdigit() and len(t) > 6:
            t = t[:6]
        out.append(t)
    return out


def embed_text(chunk: dict) -> str:
    """Текст, що індексується: заголовок документа + розділ + текст фрагмента (contextual header)."""
    return f"{chunk['title']}. {chunk['section']}. {chunk['text']}"


# ---------------------------------------------------------------- Embeddings
class Embedder:
    def __init__(self, key: str = DEFAULT_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.key = key
        self.cfg = MODELS[key]
        t0 = time.perf_counter()
        self.model = SentenceTransformer(self.cfg["name"], device=device)
        self.load_seconds = time.perf_counter() - t0
        get_dim = (getattr(self.model, "get_embedding_dimension", None)
                   or getattr(self.model, "get_sentence_embedding_dimension", None))
        self.dim = int(get_dim()) if get_dim else int(self.model.encode(["x"]).shape[1])

    @property
    def max_seq_length(self):
        return getattr(self.model, "max_seq_length", None)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def _encode(self, texts, batch_size=32):
        v = self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=len(texts) > 200)
        return np.asarray(v, dtype="float32")

    def encode_queries(self, queries):
        return self._encode([self.cfg["query_prefix"] + q for q in queries])

    def encode_passages(self, texts):
        return self._encode([self.cfg["passage_prefix"] + t for t in texts])


def cached_passage_vectors(embedder: Embedder, chunks: list[dict], tag: str):
    """Повертає (vectors, encode_seconds). Кешує на диск, щоб не перераховувати."""
    texts = [embed_text(c) for c in chunks]
    h = hashlib.sha1((embedder.cfg["name"] + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:12]
    base = CACHE / f"{embedder.key}_{tag}_{h}"
    npy, meta = base.with_suffix(".npy"), base.with_suffix(".json")
    if npy.exists() and meta.exists():
        return np.load(npy), json.loads(meta.read_text())["encode_seconds"]
    t0 = time.perf_counter()
    vecs = embedder.encode_passages(texts)
    secs = time.perf_counter() - t0
    np.save(npy, vecs)
    meta.write_text(json.dumps({"encode_seconds": secs, "n": len(texts)}))
    return vecs, secs


def pctl(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def fmt(x, nd=3):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)
