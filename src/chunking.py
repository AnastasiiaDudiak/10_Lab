"""Результат 2. Стратегії chunking і їх порівняння.

Стратегії:
  fixed        — фіксоване вікно 150 слів з overlap 30 (ігнорує структуру документа);
  structure    — за заголовками: розділ -> абзаци, пакування до 220 слів, короткі розділи (<40) зливаються з наступним;
  hierarchical — small-to-big: шукаємо за дрібними дочірніми фрагментами (60 слів, overlap 15),
                 у контекст повертаємо батьківський фрагмент (розділ до 300 слів).
Записи EM-DAT (atomic) — завжди один запис = один фрагмент.

Порівняння:  python src/chunking.py --model e5-small
Результат:   docs/chunking_cmp.md, data/chunks_<strategy>.jsonl
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, DEFAULT_MODEL, DOCS_DIR, Embedder, cached_passage_vectors, gold_subset,  # noqa: E402
                    load_gold, md_table, read_jsonl, set_seed, write_jsonl, write_text)

STRATEGIES = ["fixed", "structure", "hierarchical"]
PARAMS = {
    "fixed": dict(size=150, overlap=30),
    "structure": dict(max_words=220, min_words=40),
    "hierarchical": dict(parent_max=300, child_size=60, child_overlap=15),
}
SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯІЇЄҐ0-9«\"(])")


@dataclass
class ChunkSet:
    strategy: str
    chunks: list            # те, що індексується
    contexts: dict = field(default_factory=dict)   # context_id -> те, що повертається LLM/користувачу


def _meta(doc):
    return {k: doc[k] for k in ("doc_id", "title", "type", "date", "sensitivity", "source_url", "lang")}


def _mk(doc, strategy, n, text, units, **extra):
    c = _meta(doc)
    c.update(chunk_id=f"{doc['doc_id']}::{strategy}::{n:04d}", section=units[0].split("#", 1)[1],
             text=text, n_words=len(text.split()), units=units, strategy=strategy)
    c.update(extra)
    c.setdefault("context_id", c["chunk_id"])
    return c


def _unit(doc, section):
    return f"{doc['doc_id']}#{section}"


def _atomic(doc, strategy):
    out = []
    for i, s in enumerate(doc["sections"]):
        c = _mk(doc, strategy, i, s["text"], [_unit(doc, s["section"])])
        if s.get("date"):
            c["date"] = s["date"]
        out.append(c)
    return out


def _split_long(text, max_words):
    """Абзац довший за max_words ріжемо за реченнями."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    pieces, cur = [], []
    for sent in SENT_RE.split(text):
        sw = sent.split()
        if cur and len(cur) + len(sw) > max_words:
            pieces.append(" ".join(cur))
            cur = []
        while len(sw) > max_words:              # дуже довге речення
            pieces.append(" ".join(sw[:max_words]))
            sw = sw[max_words:]
        cur += sw
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_fixed(doc, size=150, overlap=30):
    if doc.get("atomic"):
        return _atomic(doc, "fixed")
    words, secs = [], []
    for s in doc["sections"]:
        w = s["text"].split()
        words += w
        secs += [_unit(doc, s["section"])] * len(w)
    out, i, step = [], 0, size - overlap
    while i < len(words):
        j = min(i + size, len(words))
        units = list(dict.fromkeys(secs[i:j]))
        out.append(_mk(doc, "fixed", len(out), " ".join(words[i:j]), units))
        if j == len(words):
            break
        i += step
    return out


def _structure_pieces(doc, max_words, min_words):
    """Повертає список (text, units) за структурою документа."""
    pieces, carry_text, carry_units = [], [], []
    for s in doc["sections"]:
        u = _unit(doc, s["section"])
        paras = []
        for p in s["text"].split("\n"):
            paras += _split_long(p, max_words)
        sec_words = sum(len(p.split()) for p in paras)
        if sec_words < min_words:                       # короткий розділ -> зливаємо з наступним
            carry_text += paras
            carry_units.append(u)
            continue
        cur, cur_units = list(carry_text), list(carry_units)
        carry_text, carry_units = [], []
        cur_len = sum(len(p.split()) for p in cur)
        for p in paras:
            pl = len(p.split())
            if cur and cur_len + pl > max_words and u in cur_units:
                pieces.append(("\n".join(cur), cur_units))
                cur, cur_units, cur_len = [], [], 0
            cur.append(p)
            cur_len += pl
            if u not in cur_units:
                cur_units.append(u)
        if cur:
            pieces.append(("\n".join(cur), cur_units))
    if carry_text:
        if pieces and sum(len(t.split()) for t in carry_text) < min_words:
            t, us = pieces[-1]
            pieces[-1] = (t + "\n" + "\n".join(carry_text), us + [x for x in carry_units if x not in us])
        else:
            pieces.append(("\n".join(carry_text), carry_units))
    return pieces


def chunk_structure(doc, max_words=220, min_words=40):
    if doc.get("atomic"):
        return _atomic(doc, "structure")
    return [_mk(doc, "structure", i, t, u) for i, (t, u) in enumerate(_structure_pieces(doc, max_words, min_words))]


def chunk_hierarchical(doc, parent_max=300, child_size=60, child_overlap=15):
    if doc.get("atomic"):
        ch = _atomic(doc, "hierarchical")
        return ch, {c["chunk_id"]: c for c in ch}
    parents, children = {}, []
    for pi, (ptext, punits) in enumerate(_structure_pieces(doc, parent_max, 40)):
        parent = _mk(doc, "hierarchical-parent", pi, ptext, punits)
        parents[parent["chunk_id"]] = parent
        words, i, step = ptext.split(), 0, child_size - child_overlap
        while i < len(words):
            j = min(i + child_size, len(words))
            children.append(_mk(doc, "hierarchical", len(children), " ".join(words[i:j]), punits,
                                context_id=parent["chunk_id"]))
            if j == len(words):
                break
            i += step
    return children, parents


def chunk_corpus(docs, strategy) -> ChunkSet:
    chunks, contexts = [], {}
    for d in docs:
        if strategy == "fixed":
            cs = chunk_fixed(d, **PARAMS["fixed"])
        elif strategy == "structure":
            cs = chunk_structure(d, **PARAMS["structure"])
        elif strategy == "hierarchical":
            cs, parents = chunk_hierarchical(d, **PARAMS["hierarchical"])
            contexts.update(parents)
        else:
            raise ValueError(strategy)
        chunks += cs
        if strategy != "hierarchical":
            contexts.update({c["chunk_id"]: c for c in cs})
    return ChunkSet(strategy, chunks, contexts)


def load_chunkset(strategy) -> ChunkSet:
    path = DATA / "docs.jsonl"
    if not path.exists():
        raise SystemExit("Немає data/docs.jsonl — спершу запустіть src/build_corpus.py")
    return chunk_corpus(read_jsonl(path), strategy)


# ---------------------------------------------------------------- порівняння
def main():
    from index import VectorIndex  # noqa: E402
    from metrics import contexts_from_hits, evaluate_query  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    set_seed()

    queries = gold_subset(load_gold(), "chunking")
    print(f"Тестових запитів (subset=chunking): {len(queries)}")
    emb = Embedder(args.model)
    qv = emb.encode_queries([q["query"] for q in queries])
    rows, per_query = [], {}
    for strat in STRATEGIES:
        cs = load_chunkset(strat)
        write_jsonl(DATA / f"chunks_{strat}.jsonl", cs.chunks)
        vecs, enc_s = cached_passage_vectors(emb, cs.chunks, strat)
        t0 = time.perf_counter()
        vi = VectorIndex(emb.dim)
        vi.add(vecs, cs.chunks)
        build_s = time.perf_counter() - t0
        recalls, ctx_words = [], []
        for q, v in zip(queries, qv):
            hits = vi.search(v[None, :], k=50)[0]
            ctxs = contexts_from_hits(hits, vi.meta, cs.contexts, k=args.k)
            m = evaluate_query(ctxs, q["relevant"])
            recalls.append(m["recall@5"])
            ctx_words.append(sum(c["n_words"] for c in ctxs))
            per_query.setdefault(q["qid"], {})[strat] = m["recall@5"]
        n_words = [c["n_words"] for c in cs.chunks]
        rows.append(dict(strategy=strat, params=PARAMS[strat], n_chunks=len(cs.chunks),
                         avg_chunk=sum(n_words) / len(n_words),
                         avg_ctx_item=sum(ctx_words) / len(ctx_words) / args.k,
                         ctx_words=sum(ctx_words) / len(ctx_words),
                         recall=sum(recalls) / len(recalls), enc_s=enc_s, build_s=build_s))
        print(f"{strat:13s} chunks={len(cs.chunks):5d} recall@5={rows[-1]['recall']:.3f} "
              f"контекст={rows[-1]['ctx_words']:.0f} слів")

    best = max(rows, key=lambda r: (round(r["recall"], 3), -r["ctx_words"]))
    table = md_table(
        ["Стратегія", "Параметри", "Фрагментів", "Сер. розмір фрагмента, слів", "recall@5",
         "Сер. контекст top-5, слів", "Embeddings, с", "Побудова HNSW, с"],
        [[r["strategy"], r["params"], r["n_chunks"], f"{r['avg_chunk']:.0f}", f"{r['recall']:.3f}",
          f"{r['ctx_words']:.0f}", f"{r['enc_s']:.1f}", f"{r['build_s']:.2f}"] for r in rows])
    pq = md_table(["qid", "Запит"] + STRATEGIES,
                  [[q["qid"], q["query"]] + [f"{per_query[q['qid']][s]:.2f}" for s in STRATEGIES] for q in queries])
    others = [r for r in rows if r is not best]
    reason = "; ".join(
        f"порівняно з `{r['strategy']}`: recall@5 {best['recall']:.3f} проти {r['recall']:.3f}, "
        f"контекст {best['ctx_words']:.0f} проти {r['ctx_words']:.0f} слів" for r in others)
    txt = f"""# Порівняння стратегій chunking (Результат 2)

**Модель embeddings (фіксована для всіх стратегій):** `{emb.cfg['name']}` (max_seq_length = {emb.max_seq_length} токенів).
**Індекс:** FAISS HNSW (однакові параметри). **Тестові запити:** {len(queries)} (subset `chunking` з `eval/gold.jsonl`).
**recall@5** — частка релевантних розділів (`units`), покритих першими 5 унікальними контекстами;
**розмір контексту** — сумарна кількість слів у 5 повернених контекстах (для `hierarchical` — батьківські фрагменти).

## Таблиця 2.1. Якість і розмір контексту
{table}

## Таблиця 2.2. recall@5 за запитами
{pq}

## Вибір
За даними обрано **`{best['strategy']}`**: {reason}.
Правило вибору: максимальний recall@5; за рівного recall — менший контекст (дешевше для LLM у ЛР 11).

> Примітка для захисту: розмір чанка обмежений згори max_seq_length моделі (текст понад ліміт обрізається
> й не потрапляє в embedding), знизу — втратою контексту; overlap ≈ 15–20 % зменшує ризик розрізати відповідь на межі.
"""
    write_text(DOCS_DIR / "chunking_cmp.md", txt)


if __name__ == "__main__":
    main()
