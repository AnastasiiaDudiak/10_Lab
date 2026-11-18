"""Результат 3. Векторний індекс FAISS HNSW, порівняння двох моделей embeddings, upsert/delete.

Запуск:  python src/index.py --strategy structure
Результат: docs/embeddings_cmp.md, data/index_<model>_<strategy>.faiss (+ .meta.pkl)
"""
from __future__ import annotations

import argparse
import copy
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, DOCS_DIR, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M, MODELS, SEED,  # noqa: E402
                    Embedder, cached_passage_vectors, embed_text, gold_subset, load_gold, md_table, pctl,
                    read_jsonl, set_seed, write_text)


class VectorIndex:
    """FAISS IndexHNSWFlat (косинус = inner product на нормованих векторах) + метадані.

    FAISS HNSW не підтримує remove_ids, тому:
      delete(doc)  -> позиції документа позначаються як видалені (tombstones) і відфільтровуються під час пошуку;
      upsert(doc)  -> delete старих фрагментів + add нових векторів у кінець індексу;
      compact()    -> повна перебудова без видалених (виконується періодично / коли tombstones > 20 %).
    """

    def __init__(self, dim, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, ef_search=HNSW_EF_SEARCH):
        import faiss
        self.faiss = faiss
        self.dim, self.M, self.efC, self.efS = dim, M, ef_construction, ef_search
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.meta: list[dict] = []
        self.xb = np.zeros((0, dim), dtype="float32")
        self.deleted: set[int] = set()
        self.doc_pos: dict[str, list[int]] = {}

    # ---- оновлення
    def add(self, vecs, chunks):
        vecs = np.ascontiguousarray(vecs, dtype="float32")
        start = len(self.meta)
        self.index.add(vecs)
        self.xb = np.vstack([self.xb, vecs])
        for i, c in enumerate(chunks):
            self.meta.append(c)
            self.doc_pos.setdefault(c["doc_id"], []).append(start + i)

    def delete_doc(self, doc_id) -> int:
        pos = self.doc_pos.pop(doc_id, [])
        self.deleted.update(pos)
        return len(pos)

    def upsert_doc(self, doc_id, vecs, chunks) -> int:
        n_old = self.delete_doc(doc_id)
        self.add(vecs, chunks)
        return n_old

    @property
    def n_alive(self):
        return len(self.meta) - len(self.deleted)

    def compact(self):
        keep = [i for i in range(len(self.meta)) if i not in self.deleted]
        new = VectorIndex(self.dim, self.M, self.efC, self.efS)
        new.add(self.xb[keep], [self.meta[i] for i in keep])
        self.__dict__.update(new.__dict__)

    # ---- пошук
    def search(self, qvecs, k=10, allowed=None, ef_search=None):
        """allowed: множина дозволених позицій (pre-filter за метаданими) або None.
        Повертає для кожного запиту список (pos, score)."""
        qvecs = np.ascontiguousarray(qvecs, dtype="float32")
        ef = ef_search or self.efS
        if allowed is None and not self.deleted:
            self.index.hnsw.efSearch = max(ef, k)
            D, I = self.index.search(qvecs, k)
            return [[(int(i), float(d)) for d, i in zip(dr, ir) if i >= 0] for dr, ir in zip(D, I)]
        ok = set(range(len(self.meta))) if allowed is None else set(allowed)
        ok -= self.deleted
        if not ok:
            return [[] for _ in qvecs]
        ids = np.fromiter(sorted(ok), dtype="int64")
        if len(ids) < 0.2 * len(self.meta) or len(ids) <= k:
            # дуже вибірковий фільтр: точний пошук по підмножині (HNSW з жорстким фільтром губить сусідів)
            scores = qvecs @ self.xb[ids].T
            out = []
            for row in scores:
                top = np.argsort(-row)[:k]
                out.append([(int(ids[j]), float(row[j])) for j in top])
            return out
        try:
            sel = self.faiss.IDSelectorBatch(ids)
        except TypeError:
            sel = self.faiss.IDSelectorBatch(len(ids), self.faiss.swig_ptr(ids))
        try:
            params = self.faiss.SearchParametersHNSW()
            params.efSearch = max(ef, k)
            params.sel = sel
            D, I = self.index.search(qvecs, k, params=params)
            return [[(int(i), float(d)) for d, i in zip(dr, ir) if i >= 0] for dr, ir in zip(D, I)]
        except Exception:      # старі версії faiss без SearchParameters: post-filter з запасом
            self.index.hnsw.efSearch = max(ef, k * 10)
            D, I = self.index.search(qvecs, min(len(self.meta), k * 10))
            return [[(int(i), float(d)) for d, i in zip(dr, ir) if i in ok][:k] for dr, ir in zip(D, I)]

    # ---- збереження
    def save(self, path):
        path = Path(path)
        self.faiss.write_index(self.index, str(path))
        with open(path.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump({k: v for k, v in self.__dict__.items() if k not in ("index", "faiss")}, f)

    @classmethod
    def load(cls, path):
        import faiss
        path = Path(path)
        obj = cls.__new__(cls)
        with open(path.with_suffix(".meta.pkl"), "rb") as f:
            obj.__dict__.update(pickle.load(f))
        obj.faiss, obj.index = faiss, faiss.read_index(str(path))
        return obj


def build_index(emb: Embedder, chunks, tag):
    vecs, enc_s = cached_passage_vectors(emb, chunks, tag)
    t0 = time.perf_counter()
    vi = VectorIndex(emb.dim)
    vi.add(vecs, chunks)
    return vi, enc_s, time.perf_counter() - t0


# ---------------------------------------------------------------- upsert/delete
def update_test(emb: Embedder, vi: VectorIndex, cs, docs):
    """Перевірка процедури оновлення на 5 документах."""
    from chunking import chunk_corpus
    rng = np.random.default_rng(SEED)
    cand = [d for d in docs if d["type"] == "article" and len(d["sections"]) >= 3]
    picked = [cand[i] for i in sorted(rng.choice(len(cand), size=min(5, len(cand)), replace=False))]
    rows = []

    def doc_hits(query, doc_id, k=20, within_doc=False):
        qv = emb.encode_queries([query])
        allowed = set(vi.doc_pos.get(doc_id, [])) if within_doc else None
        if within_doc and not allowed:
            return []
        hits = vi.search(qv, k=k, allowed=allowed)[0]
        return [vi.meta[p] for p, _ in hits if vi.meta[p]["doc_id"] == doc_id]

    def chunks_of(doc):
        return chunk_corpus([doc], cs.strategy).chunks

    def encode(chs):
        return emb.encode_passages([embed_text(c) for c in chs])

    # 1. delete
    d = picked[0]
    before = len(vi.doc_pos.get(d["doc_id"], []))
    vi.delete_doc(d["doc_id"])
    found = doc_hits(d["title"], d["doc_id"])
    rows.append([d["doc_id"], d["title"], "delete", before, len(vi.doc_pos.get(d["doc_id"], [])),
                 "пошук за назвою не повертає фрагментів документа", "PASS" if not found else "FAIL"])
    # 2. upsert зі зміненим текстом (додано новий абзац із унікальним маркером)
    d = copy.deepcopy(picked[1])
    marker = f"Тестове оновлення LAB10: документ {d['doc_id']} доповнено протоколом перевірки оповіщення населення."
    d["sections"][0]["text"] += "\n" + marker
    old_pos = list(vi.doc_pos.get(d["doc_id"], []))
    before = len(old_pos)
    new = chunks_of(d)
    vi.upsert_doc(d["doc_id"], encode(new), new)
    found = doc_hits(marker, d["doc_id"], k=3, within_doc=True)       # пошук у межах документа
    ok = any("LAB10" in c["text"] for c in found) and all(p in vi.deleted for p in old_pos)
    rows.append([d["doc_id"], d["title"], "upsert (змінено текст)", before, len(vi.doc_pos[d["doc_id"]]),
                 "новий текст — у top-3 пошуку в межах документа; старі версії видалено", "PASS" if ok else "FAIL"])
    # 3. upsert без змін (повторна індексація) — кількість живих фрагментів не змінюється
    d = picked[2]
    alive0, before = vi.n_alive, len(vi.doc_pos.get(d["doc_id"], []))
    new = chunks_of(d)
    vi.upsert_doc(d["doc_id"], encode(new), new)
    ok = vi.n_alive == alive0 and len(vi.doc_pos[d["doc_id"]]) == before
    rows.append([d["doc_id"], d["title"], "upsert (без змін)", before, len(vi.doc_pos[d["doc_id"]]),
                 "кількість живих фрагментів не змінилась (ідемпотентність)", "PASS" if ok else "FAIL"])
    # 4. delete + повторне додавання
    d = picked[3]
    before = len(vi.doc_pos.get(d["doc_id"], []))
    vi.delete_doc(d["doc_id"])
    gone = not doc_hits(d["title"], d["doc_id"])
    new = chunks_of(d)
    vi.upsert_doc(d["doc_id"], encode(new), new)
    back = bool(doc_hits(d["title"], d["doc_id"]))
    rows.append([d["doc_id"], d["title"], "delete → add", before, len(vi.doc_pos[d["doc_id"]]),
                 "після delete не знаходиться, після add — знаходиться", "PASS" if gone and back else "FAIL"])
    # 5. upsert з видаленим розділом
    d = copy.deepcopy(picked[4])
    removed = d["sections"].pop()
    unit = f"{d['doc_id']}#{removed['section']}"
    before = len(vi.doc_pos.get(d["doc_id"], []))
    new = chunks_of(d)
    vi.upsert_doc(d["doc_id"], encode(new), new)
    found = doc_hits(removed["text"][:300], d["doc_id"], k=20)
    ok = all(unit not in c["units"] for c in found)
    rows.append([d["doc_id"], d["title"], f"upsert (видалено розділ «{removed['section']}»)", before,
                 len(vi.doc_pos[d["doc_id"]]), "фрагменти видаленого розділу більше не повертаються",
                 "PASS" if ok else "FAIL"])
    tomb = len(vi.deleted)
    vi.compact()
    rows.append(["—", "весь індекс", "compact()", f"tombstones: {tomb}", f"ntotal={vi.index.ntotal}",
                 "після перебудови ntotal = кількість живих фрагментів",
                 "PASS" if vi.index.ntotal == vi.n_alive and not vi.deleted else "FAIL"])
    # чи підтримує HNSW remove_ids (обґрунтування tombstones)
    try:
        vi.index.remove_ids(np.array([0], dtype="int64"))
        remove_note = "remove_ids спрацював (ця версія faiss підтримує видалення)"
    except Exception as e:  # noqa: BLE001
        remove_note = f"remove_ids → помилка: `{str(e).splitlines()[0][:120]}`"
    return rows, remove_note


# ---------------------------------------------------------------- main
def main():
    from chunking import load_chunkset
    from metrics import contexts_from_hits, evaluate_query

    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    set_seed()

    cs = load_chunkset(args.strategy)
    docs = read_jsonl(DATA / "docs.jsonl")
    queries = gold_subset(load_gold(), "embeddings")
    print(f"Фрагментів: {len(cs.chunks)}; запитів (subset=embeddings): {len(queries)}")

    res, ef_rows, saved = [], [], {}
    for key in args.models:
        emb = Embedder(key)
        vi, enc_s, build_s = build_index(emb, cs.chunks, args.strategy)
        vi.save(DATA / f"index_{key}_{args.strategy}.faiss")
        q_enc, q_search, rec, mrrs, by_lang = [], [], [], [], {}
        emb.encode_queries(["розігрів"])   # warm-up
        for q in queries:
            t0 = time.perf_counter()
            qv = emb.encode_queries([q["query"]])
            t1 = time.perf_counter()
            hits = vi.search(qv, k=50)[0]
            t2 = time.perf_counter()
            m = evaluate_query(contexts_from_hits(hits, vi.meta, cs.contexts, 10), q["relevant"])
            q_enc.append((t1 - t0) * 1000)
            q_search.append((t2 - t1) * 1000)
            rec.append(m["recall@5"])
            mrrs.append(m["MRR"])
            by_lang.setdefault(q.get("lang", "uk"), []).append(m["recall@5"])
        res.append(dict(key=key, name=emb.cfg["name"], dim=emb.dim, seq=emb.max_seq_length,
                        params=emb.n_params(), enc_s=enc_s, build_s=build_s, rec=np.mean(rec),
                        mrr=np.mean(mrrs), by_lang={k: np.mean(v) for k, v in by_lang.items()},
                        n_lang={k: len(v) for k, v in by_lang.items()},
                        enc_med=pctl(q_enc, 50), enc_p95=pctl(q_enc, 95),
                        s_med=pctl(q_search, 50), s_p95=pctl(q_search, 95)))
        # efSearch: якість vs затримка
        qv_all = emb.encode_queries([q["query"] for q in queries])
        for ef in (16, 32, 64, 128):
            t0 = time.perf_counter()
            all_hits = vi.search(qv_all, k=50, ef_search=ef)
            ms = (time.perf_counter() - t0) * 1000 / len(queries)
            r = np.mean([evaluate_query(contexts_from_hits(h, vi.meta, cs.contexts, 10), q["relevant"])["recall@5"]
                         for h, q in zip(all_hits, queries)])
            ef_rows.append([key, ef, f"{r:.3f}", f"{ms:.3f}"])
        saved[key] = (emb, vi)
        print(f"{key}: recall@5={res[-1]['rec']:.3f} MRR={res[-1]['mrr']:.3f} "
              f"побудова={enc_s + build_s:.1f}s пошук(мед)={res[-1]['s_med']:.2f}ms")

    best = max(res, key=lambda r: (round(r["rec"], 3), r["mrr"]))
    langs = sorted({l for r in res for l in r["by_lang"]})
    t_params = md_table(["Модель", "Розмірність", "max_seq_length", "Параметрів, млн", "M", "efConstruction",
                         "efSearch", "Метрика", "Час embeddings корпусу, с", "Час побудови HNSW, с"],
                        [[r["name"], r["dim"], r["seq"], f"{r['params'] / 1e6:.0f}", HNSW_M, HNSW_EF_CONSTRUCTION,
                          HNSW_EF_SEARCH, "cosine (IP)", f"{r['enc_s']:.1f}", f"{r['build_s']:.3f}"] for r in res])
    t_q = md_table(["Модель", "recall@5", "MRR@10"] + [f"recall@5 ({l}, n={res[0]['n_lang'].get(l, 0)})" for l in langs]
                   + ["Кодування запиту, мс (мед / p95)", "Пошук HNSW, мс (мед / p95)"],
                   [[r["key"], f"{r['rec']:.3f}", f"{r['mrr']:.3f}"]
                    + [fmt_lang(r, l) for l in langs]
                    + [f"{r['enc_med']:.1f} / {r['enc_p95']:.1f}", f"{r['s_med']:.2f} / {r['s_p95']:.2f}"] for r in res])
    t_ef = md_table(["Модель", "efSearch", "recall@5", "Пошук, мс/запит"], ef_rows)

    emb, vi = saved[best["key"]]
    upd_rows, remove_note = update_test(emb, vi, cs, docs)
    t_upd = md_table(["doc_id", "Документ", "Операція", "Фрагментів до", "Фрагментів після", "Перевірка", "Результат"],
                     upd_rows)

    txt = f"""# Векторний індекс і порівняння embeddings (Результат 3)

**Стратегія chunking:** `{args.strategy}` ({len(cs.chunks)} фрагментів). **Індекс:** FAISS `IndexHNSWFlat`, inner product
на L2-нормованих векторах (= косинусна схожість). **Запити:** {len(queries)} (subset `embeddings`).
Для e5 використано префікси `query:` / `passage:`. Seed = {SEED}.

## Таблиця 3.1. Моделі, параметри індексу та час побудови
{t_params}

## Таблиця 3.2. Якість і затримка на {len(queries)} запитах
{t_q}

## Таблиця 3.3. Вплив efSearch (компроміс якість/швидкість)
{t_ef}

**Висновок.** Краща модель за recall@5 — **`{best['name']}`**. Мова: у таблиці 3.2 recall окремо для українських
та англійських запитів (та різномовних пар запит/документ). Варто звернути увагу на max_seq_length:
текст фрагмента понад ліміт обрізається.

## Процедура оновлення індексу (upsert / delete)
1. Документ змінено → перечанкувати лише цей документ тією ж стратегією.
2. `delete_doc(doc_id)`: усі позиції документа потрапляють у множину tombstones і відфільтровуються під час пошуку
   (FAISS HNSW не підтримує фізичне видалення: {remove_note}).
3. `upsert_doc(doc_id)`: delete старої версії + `add` нових векторів (нові позиції в кінці графа).
4. `compact()`: періодична повна перебудова без tombstones (напр., коли видалених > 20 %).
5. Метадані (`doc_pos`, `meta`) зберігаються поруч з індексом (`.meta.pkl`).

## Таблиця 3.4. Перевірка на 5 документах (модель `{best['key']}`)
{t_upd}
"""
    write_text(DOCS_DIR / "embeddings_cmp.md", txt)
    print("\n".join(" | ".join(map(str, r)) for r in upd_rows))


def fmt_lang(r, l):
    v = r["by_lang"].get(l)
    return "—" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
