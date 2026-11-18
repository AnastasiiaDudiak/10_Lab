"""Результат 5. Оцінювання конфігурацій на золотому наборі: recall@5/10, MRR, nDCG@10, затримка, вартість.

Запуск:  python src/evaluate.py --strategy structure --model e5-small
Результат: docs/search_eval.md, eval/runs/<config>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_MODEL, DOCS_DIR, EVAL, RERANKER, load_gold, md_table, pctl, set_seed, write_jsonl, write_text  # noqa: E402

CONFIGS = ["bm25_nostem", "bm25", "dense", "hybrid", "hybrid_rerank"]
METRICS = ["recall@5", "recall@10", "MRR", "nDCG@10"]


def main():
    from hybrid import HybridSearcher
    from metrics import evaluate_query
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    set_seed()

    gold = load_gold()
    ans = [g for g in gold if g.get("relevant")]
    noans = [g for g in gold if not g.get("relevant")]
    qtypes = Counter(g.get("qtype") for g in gold)
    print(f"Запитів: {len(gold)} (з відповіддю {len(ans)}, без відповіді {len(noans)}); типи: {dict(qtypes)}")

    hs = HybridSearcher(args.strategy, args.model)
    hs_ns = HybridSearcher.__new__(HybridSearcher)      # BM25 без стемінгу (для перевірки гіпотези)
    from rank_bm25 import BM25Okapi

    from common import bm25_tokens, embed_text
    hs_ns.__dict__.update(hs.__dict__)
    hs_ns.stem = False
    hs_ns.bm25 = BM25Okapi([bm25_tokens(embed_text(c), False) for c in hs.chunks])

    results, lat, by_type = {}, defaultdict(list), defaultdict(lambda: defaultdict(list))
    for cfg in CONFIGS:
        searcher, mode = (hs_ns, "bm25") if cfg == "bm25_nostem" else (hs, cfg)
        searcher.search("розігрів", mode)                      # warm-up
        runs, ms_all = [], []
        for g in gold:
            ctxs, ms = searcher.search(g["query"], mode)
            ms_all.append(ms)
            m = evaluate_query(ctxs, g["relevant"]) if g.get("relevant") else {}
            runs.append({"qid": g["qid"], "ms": ms, "metrics": m,
                         "top": [{"chunk_id": c["chunk_id"], "units": c["units"], "score": c["score"]} for c in ctxs]})
            if m:
                by_type[cfg][g.get("qtype")].append(m["nDCG@10"])
        write_jsonl(EVAL / "runs" / f"{cfg}.jsonl", runs)
        results[cfg] = {k: float(np.mean([r["metrics"][k] for r in runs if r["metrics"]])) for k in METRICS}
        lat[cfg] = (pctl(ms_all, 50), pctl(ms_all, 95))
        print(f"{cfg:14s} " + " ".join(f"{k}={results[cfg][k]:.3f}" for k in METRICS)
              + f"  latency med={lat[cfg][0]:.0f}ms")

    # ---- запити без відповіді: чи можна відсікти за score reranker-а
    def top_ce(g):
        ctxs, _ = hs.search(g["query"], "hybrid_rerank", k=1)
        return ctxs[0]["score"] if ctxs else -1e9
    s_ans = [top_ce(g) for g in ans]
    s_no = [top_ce(g) for g in noans]
    best_thr, best_bacc = None, -1
    for thr in sorted(set(s_ans + s_no)):
        tpr = np.mean([s >= thr for s in s_ans])
        tnr = np.mean([s < thr for s in s_no]) if s_no else 0
        bacc = (tpr + tnr) / 2
        if bacc > best_bacc:
            best_thr, best_bacc = thr, bacc

    # ---- κ (якщо вже пораховано)
    kfile = EVAL / "kappa.json"
    kappa_txt = "_κ ще не обчислено: `python src/gold_tools.py kappa`_"
    if kfile.exists():
        k = json.loads(kfile.read_text(encoding="utf-8"))
        kappa_txt = (f"Розмітники: {k.get('annotators', 'A, B')}. **{k['n_queries']} запитів, {k['n_pairs']} пар (запит, фрагмент)**: "
                     f"p_o = {k['po']:.3f}, p_e = {k['pe']:.3f}, **κ Коена = {k['kappa']:.3f}** ({k['interpretation']}).")

    # ---- вартість: параметри моделей
    n_emb = hs.emb.n_params()
    n_ce = int(sum(p.numel() for p in getattr(hs.ce, "model", hs.ce).parameters())) if hs.ce else 0

    best = max(CONFIGS, key=lambda c: results[c]["nDCG@10"])
    t_main = md_table(["Конфігурація"] + METRICS + ["Затримка, мс (мед / p95)"],
                      [[c] + [f"{results[c][k]:.3f}" for k in METRICS] + [f"{lat[c][0]:.0f} / {lat[c][1]:.0f}"]
                       for c in CONFIGS])
    types = [t for t in ["keyword", "paraphrase", "multihop"] if t in qtypes]
    t_type = md_table(["Конфігурація"] + [f"nDCG@10 · {t} (n={len(by_type[CONFIGS[0]][t])})" for t in types],
                      [[c] + [f"{np.mean(by_type[c][t]):.3f}" for t in types] for c in CONFIGS])
    txt = f"""# Золотий набір і оцінювання (Результат 5)

## 5.1. Золотий набір (`eval/gold.jsonl`)
Усього **{len(gold)}** запитів: {', '.join(f'{k}: {v}' for k, v in qtypes.items())}; без відповіді — **{len(noans)}**.
Релевантність розмічено на рівні розділів (`doc_id#section`), оцінки 2 — пряма відповідь, 1 — часткова.
Кандидати запитів запропоновано LLM (див. `llm_log/`), кожен перевірено й виправлено вручну (рівень V3).

## 5.2. Узгодженість розмітників
{kappa_txt}

## 5.3. Метрики конфігурацій (запити з відповіддю, n = {len(ans)}; стратегія `{args.strategy}`, модель `{hs.emb.cfg['name']}`)
{t_main}

`bm25_nostem` — BM25 без обрізання слів (перевірка гіпотези, що «легкий стемінг» допомагає для української).

## 5.4. nDCG@10 за типами запитів
{t_type}

## 5.5. Запити без відповіді
Максимальний score cross-encoder: з відповіддю — медіана {np.median(s_ans):.2f}; без відповіді — медіана
{np.median(s_no) if s_no else float('nan'):.2f}. Поріг {best_thr:.2f} дає balanced accuracy {best_bacc:.2f}
у розпізнаванні «відповіді в корпусі немає» (для ЛР 11: якщо top-1 < порога — відповідати «не знайдено»).

## 5.6. Вартість і затримка
- Усі моделі працюють локально на CPU: **вартість API = 0 $**, дані не передаються в зовнішні сервіси.
- Embedding-модель: {n_emb / 1e6:.0f} млн параметрів; reranker: {n_ce / 1e6:.0f} млн параметрів (`{RERANKER}`).
- Затримка: див. таблицю 5.3 (reranker додає приблизно {lat['hybrid_rerank'][0] - lat['hybrid'][0]:.0f} мс на запит,
  бо cross-encoder обробляє 30 пар «запит–фрагмент»).

## 5.7. Висновок (вибір для ЛР 11)
Найвищий nDCG@10 — **`{best}`** ({results[best]['nDCG@10']:.3f}).
_TODO: сформулювати вибір з урахуванням якості, затримки та вартості (заповнюється після запуску)._
"""
    write_text(DOCS_DIR / "search_eval.md", txt)


if __name__ == "__main__":
    main()
