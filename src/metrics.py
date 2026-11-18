"""Метрики пошуку: recall@k, MRR, nDCG@k.

Релевантність розмічено на рівні розділів документа (unit = "doc_id#section") з оцінкою 1 (частково)
або 2 (пряма відповідь). Фрагмент вважається релевантним, якщо покриває хоча б один релевантний unit.
Так одна розмітка працює для всіх стратегій chunking.
  recall@k = |релевантні units, покриті top-k| / |усі релевантні units|
  MRR      = 1 / ранг першого релевантного контексту (0, якщо в top-10 немає)
  nDCG@k   = DCG / IDCG, gain = 2^g − 1, кожен unit зараховується лише один раз (перший раз, коли з'явився)
"""
from __future__ import annotations

import math


def contexts_from_hits(hits, meta, contexts, k=10):
    """hits: [(pos, score)] у порядку рангу -> перші k унікальних контекстів (для hierarchical — батьки)."""
    out, seen = [], set()
    for pos, score in hits:
        ch = meta[pos]
        cid = ch.get("context_id", ch["chunk_id"])
        if cid in seen:
            continue
        seen.add(cid)
        ctx = dict(contexts.get(cid, ch))
        ctx["score"] = float(score)
        out.append(ctx)
        if len(out) >= k:
            break
    return out


def recall_at_k(ranked_units, rel, k):
    if not rel:
        return float("nan")
    covered = set()
    for units in ranked_units[:k]:
        covered |= set(units) & rel.keys()
    return len(covered) / len(rel)


def mrr(ranked_units, rel, k=10):
    for i, units in enumerate(ranked_units[:k], 1):
        if set(units) & rel.keys():
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_units, rel, k=10):
    if not rel:
        return float("nan")
    seen, dcg = set(), 0.0
    for i, units in enumerate(ranked_units[:k], 1):
        new = (set(units) & rel.keys()) - seen
        g = max((rel[u] for u in new), default=0)
        seen |= new
        dcg += (2 ** g - 1) / math.log2(i + 1)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def first_relevant_rank(ranked_units, rel, k=30):
    for i, units in enumerate(ranked_units[:k], 1):
        if set(units) & rel.keys():
            return i
    return None


def evaluate_query(contexts, rel):
    ru = [c["units"] for c in contexts]
    return {"recall@5": recall_at_k(ru, rel, 5), "recall@10": recall_at_k(ru, rel, 10),
            "MRR": mrr(ru, rel, 10), "nDCG@10": ndcg_at_k(ru, rel, 10),
            "first_rel": first_relevant_rank(ru, rel)}


def cohen_kappa(a, b):
    """Каппа Коена для двох розмітників (довільні мітки)."""
    assert len(a) == len(b) and a
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po, pe
