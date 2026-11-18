"""Результат 4. Гібридний пошук: BM25 + dense, злиття RRF (k=60), reranker (cross-encoder), pre-filter.

Запуск:  python src/hybrid.py --strategy structure --model e5-small
         python src/hybrid.py --query "як евакуювати населення під час повені" --type article --sensitivity public
Результат: docs/hybrid_report.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_MODEL, DOCS_DIR, RERANKER, Embedder, bm25_tokens, cached_passage_vectors,  # noqa: E402
                    embed_text, load_gold, md_table, set_seed, write_text)

MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
SENS_LEVELS = {"public": ["public"], "internal": ["public", "internal"],
               "restricted": ["public", "internal", "restricted"]}


def rrf(rank_lists, k=60, top=None):
    """Reciprocal Rank Fusion: score(d) = Σ_i 1 / (k + rank_i(d))."""
    scores = {}
    for lst in rank_lists:
        for rank, (pos, _) in enumerate(lst, 1):
            scores[pos] = scores.get(pos, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top] if top else fused


class HybridSearcher:
    def __init__(self, strategy="structure", model_key=DEFAULT_MODEL, use_reranker=True, stem=True):
        from rank_bm25 import BM25Okapi

        from chunking import load_chunkset
        from index import VectorIndex
        self.cs = load_chunkset(strategy)
        self.chunks = self.cs.chunks
        self.stem = stem
        self.bm25 = BM25Okapi([bm25_tokens(embed_text(c), stem) for c in self.chunks])
        self.emb = Embedder(model_key)
        vecs, _ = cached_passage_vectors(self.emb, self.chunks, strategy)
        self.vi = VectorIndex(self.emb.dim)
        self.vi.add(vecs, self.chunks)
        self.ce = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.ce = CrossEncoder(RERANKER, max_length=512)

    # ---------- pre-filter
    def allowed(self, types=None, sensitivity=None, date_from=None, date_to=None):
        """Повертає множину дозволених позицій або None (без фільтра).
        sensitivity — рівень доступу користувача: public | internal | restricted."""
        if not any([types, sensitivity, date_from, date_to]):
            return None
        levels = SENS_LEVELS[sensitivity] if sensitivity else None
        ok = set()
        for i, c in enumerate(self.chunks):
            if types and c["type"] not in types:
                continue
            if levels and c["sensitivity"] not in levels:
                continue
            d = c.get("date") or ""
            if date_from and (not d or d < date_from):
                continue
            if date_to and (not d or d > date_to):
                continue
            ok.add(i)
        return ok

    # ---------- окремі ретривери
    def bm25_search(self, query, k=50, allowed=None):
        scores = self.bm25.get_scores(bm25_tokens(query, self.stem))
        if allowed is not None:
            mask = np.full(len(scores), -np.inf)
            idx = list(allowed)
            mask[idx] = scores[idx]
            scores = mask
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def dense_search(self, query, k=50, allowed=None):
        qv = self.emb.encode_queries([query])
        return self.vi.search(qv, k=k, allowed=allowed)[0]

    def rerank(self, query, cand, top=None):
        texts = [embed_text(self.chunks[p]) for p, _ in cand]
        if not texts:
            return []
        scores = self.ce.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)
        out = sorted(zip([p for p, _ in cand], map(float, scores)), key=lambda x: -x[1])
        return out[:top] if top else out

    # ---------- єдиний інтерфейс
    def search(self, query, mode="hybrid_rerank", k=10, filters=None, n_cand=50, n_rerank=30):
        from metrics import contexts_from_hits
        allowed = self.allowed(**(filters or {}))
        t0 = time.perf_counter()
        if mode in ("bm25", "bm25_nostem"):
            hits = self.bm25_search(query, n_cand, allowed)
        elif mode == "dense":
            hits = self.dense_search(query, n_cand, allowed)
        elif mode in ("hybrid", "hybrid_rerank"):
            hits = rrf([self.bm25_search(query, n_cand, allowed), self.dense_search(query, n_cand, allowed)], k=60)
            if mode == "hybrid_rerank":
                hits = self.rerank(query, hits[:n_rerank])        # топ-30 -> переранжування
        else:
            raise ValueError(mode)
        ms = (time.perf_counter() - t0) * 1000
        return contexts_from_hits(hits, self.vi.meta, self.cs.contexts, k), ms


def show(ctxs, n=5, width=90):
    lines = []
    for i, c in enumerate(ctxs[:n], 1):
        lines.append(f"  {i}. [{c['score']:.4f}] {c['title']} › {c['section']}  ({c['type']}, {c['sensitivity']}, "
                     f"{c.get('date', '')})\n     {c['text'][:width].replace(chr(10), ' ')}…")
    return "\n".join(lines) or "  (порожньо)"


def ctx_table(ctxs, n=5):
    return md_table(["#", "Score", "Документ › розділ", "type", "sensitivity", "date"],
                    [[i, f"{c['score']:.4f}", f"{c['title']} › {c['section']}", c["type"], c["sensitivity"],
                      c.get("date", "")] for i, c in enumerate(ctxs[:n], 1)])


def main():
    from metrics import evaluate_query
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--query")
    ap.add_argument("--mode", default="hybrid_rerank", choices=MODES)
    ap.add_argument("--type", nargs="*")
    ap.add_argument("--sensitivity", choices=list(SENS_LEVELS))
    ap.add_argument("--date-from")
    ap.add_argument("--date-to")
    args = ap.parse_args()
    set_seed()
    hs = HybridSearcher(args.strategy, args.model)

    if args.query:      # інтерактивний режим для демонстрації / захисту
        filters = dict(types=args.type, sensitivity=args.sensitivity, date_from=args.date_from, date_to=args.date_to)
        modes = MODES if args.mode == "hybrid_rerank" else [args.mode]
        for m in modes:
            ctxs, ms = hs.search(args.query, m, filters=filters)
            print(f"\n=== {m} ({ms:.0f} мс) ===\n{show(ctxs)}")
        return

    gold = [g for g in load_gold() if g.get("relevant")]
    # ---- C1/C2: демонстрація на першому запиті
    demo = gold[0]
    demo_out = {}
    for m in MODES:
        ctxs, ms = hs.search(demo["query"], m)
        demo_out[m] = (ctxs, ms)
        print(f"\n=== {m} ({ms:.0f} мс) — «{demo['query']}» ===\n{show(ctxs)}")

    # ---- оцінка всіх відповідних запитів для прикладів і зміни порядку reranker-ом
    per_q, changed = [], 0
    for g in gold:
        row = {"g": g}
        for m in MODES:
            ctxs, ms = hs.search(g["query"], m)
            row[m] = (ctxs, evaluate_query(ctxs, g["relevant"]))
        top_h = [c["chunk_id"] for c in row["hybrid"][0][:5]]
        top_r = [c["chunk_id"] for c in row["hybrid_rerank"][0][:5]]
        changed += top_h != top_r
        per_q.append(row)
    print(f"\nReranker змінив топ-5 у {changed} з {len(gold)} запитів")

    # ---- C3: pre-filter
    pf_cases = [
        ("роль public: sensitivity=public", dict(sensitivity="public"),
         lambda c: c["sensitivity"] == "public"),
        ("лише записи EM-DAT: type=record (роль restricted)", dict(types=["record"], sensitivity="restricted"),
         lambda c: c["type"] == "record"),
        ("дата ≤ 2023-12-31 (події 2023 р., старі редакції)", dict(date_to="2023-12-31"),
         lambda c: "" < (c.get("date") or "") <= "2023-12-31"),
        ("порожній фільтр-перевірка: type=article і дата ≤ 1900-01-01", dict(types=["article"], date_to="1900-01-01"),
         lambda c: False),
    ]
    pf_query = demo["query"]
    pf_rows, pf_details = [], []
    for name, flt, check in pf_cases:
        n_allowed = len(hs.allowed(**flt))
        ctxs, ms = hs.search(pf_query, "hybrid_rerank", filters=flt)
        ok = all(check(c) for c in ctxs)
        pf_rows.append([name, n_allowed, len(ctxs), "PASS" if ok else "FAIL", f"{ms:.0f}"])
        pf_details.append(f"**{name}**\n\n{ctx_table(ctxs)}\n")
        print(f"pre-filter «{name}»: дозволено {n_allowed}, повернуто {len(ctxs)} — {'PASS' if ok else 'FAIL'}")

    # ---- C4: приклади виграшу/програшу hybrid vs dense (за MRR, далі nDCG)
    def delta(r):
        return (r["hybrid"][1]["MRR"] - r["dense"][1]["MRR"], r["hybrid"][1]["nDCG@10"] - r["dense"][1]["nDCG@10"])
    wins = sorted([r for r in per_q if delta(r) > (0, 0)], key=delta, reverse=True)[:3]
    losses = sorted([r for r in per_q if delta(r) < (0, 0)], key=delta)[:1]

    def ex_block(r, kind):
        g = r["g"]
        ranks = {m: r[m][1]["first_rel"] for m in MODES}
        rk = ", ".join(f"{m}: {ranks[m] if ranks[m] else '>10'}" for m in MODES)
        return f"""### {kind}: {g['qid']} — «{g['query']}» (тип: {g.get('qtype')})
Ранг першого релевантного: {rk}. MRR dense = {r['dense'][1]['MRR']:.3f}, hybrid = {r['hybrid'][1]['MRR']:.3f};
nDCG@10 dense = {r['dense'][1]['nDCG@10']:.3f}, hybrid = {r['hybrid'][1]['nDCG@10']:.3f}.
Релевантні розділи: {', '.join(g['relevant'])}

Dense, топ-3:
{ctx_table(r['dense'][0], 3)}

Hybrid (RRF), топ-3:
{ctx_table(r['hybrid'][0], 3)}

**Пояснення:** _TODO — чому так (точні терміни/назви/числа → BM25; перефразування → dense)._
"""

    demo_tables = "\n".join(f"### {m} ({demo_out[m][1]:.0f} мс)\n{ctx_table(demo_out[m][0])}\n" for m in MODES)
    txt = f"""# Гібридний пошук і reranking (Результат 4)

**Конфігурація:** стратегія `{args.strategy}`, dense = `{hs.emb.cfg['name']}` (FAISS HNSW), BM25 = `rank_bm25.BM25Okapi`
(k1=1.5, b=0.75; токенізація: нижній регістр, стоп-слова, обрізання слів до 6 символів), злиття **RRF, k = 60**
(кандидати: топ-50 з кожного ретривера), reranker = `{RERANKER}` (cross-encoder) для **топ-30 → топ-5**.

Схема: `запит → pre-filter (type, sensitivity, date) → [BM25 top-50] + [dense top-50] → RRF → top-30 → cross-encoder → top-5`

## 4.1. Робота BM25, dense, RRF і reranker (запит «{demo['query']}»)
{demo_tables}
**Вплив reranker:** топ-5 після переранжування відрізняється від топ-5 RRF у **{changed} з {len(gold)}** запитів.

## 4.2. Pre-filter за метаданими (запит «{pf_query}», режим hybrid_rerank)
Фільтр застосовується **до** пошуку: BM25 рахується лише для дозволених фрагментів, у FAISS передається
`IDSelectorBatch` (для дуже вибіркових фільтрів — точний пошук по підмножині).

{md_table(["Фільтр", "Дозволено фрагментів", "Повернуто", "Усі результати задовольняють фільтр", "мс"], pf_rows)}

{chr(10).join(pf_details)}
## 4.3. Де гібридний пошук виграє і де програє (hybrid RRF vs dense)
{chr(10).join(ex_block(r, f'Виграш {i}') for i, r in enumerate(wins, 1)) or '_немає запитів, де hybrid кращий_'}
{chr(10).join(ex_block(r, 'Програш') for r in losses) or '_немає запитів, де hybrid гірший_'}
"""
    write_text(DOCS_DIR / "hybrid_report.md", txt)


if __name__ == "__main__":
    main()
