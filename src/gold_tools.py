"""Інструменти для золотого набору.

  python src/gold_tools.py check                 — перевірка eval/gold.jsonl (≥40 запитів, ≥5 без відповіді, units існують)
  python src/gold_tools.py review                — eval/gold_review.csv: запит + релевантні розділи з текстом (для ручної перевірки V3)
  python src/gold_tools.py pool --n 10           — eval/annotation_pool.csv: пул кандидатів (top-5 з 4 конфігурацій) для 2 розмітників
  python src/gold_tools.py kappa                 — κ Коена за заповненими колонками label_A / label_B -> eval/kappa.json, eval/kappa.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DEFAULT_MODEL, EVAL, load_gold, md_table, read_jsonl, write_text  # noqa: E402


def all_units():
    units = {}
    for d in read_jsonl(DATA / "docs.jsonl"):
        for s in d["sections"]:
            units[f"{d['doc_id']}#{s['section']}"] = (d["title"], s["section"], s["text"])
    return units


def cmd_check(_):
    gold, units = load_gold(), all_units()
    qids = Counter(g["qid"] for g in gold)
    noans = [g for g in gold if not g.get("relevant")]
    bad = [(g["qid"], u) for g in gold for u in g.get("relevant", {}) if u not in units]
    subs = Counter(s for g in gold for s in g.get("subsets", []))
    print(f"Запитів: {len(gold)} ({'OK' if len(gold) >= 40 else '<40!'}); без відповіді: {len(noans)} "
          f"({'OK' if len(noans) >= 5 else '<5!'})")
    print(f"Типи: {dict(Counter(g.get('qtype') for g in gold))}; мови: {dict(Counter(g.get('lang') for g in gold))}")
    print(f"Підмножини: {dict(subs)} (потрібно chunking ≥10, embeddings ≥20)")
    print(f"Дублікати qid: {[q for q, n in qids.items() if n > 1] or 'немає'}")
    print(f"Неіснуючі units: {bad or 'немає'}")


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:      # utf-8-sig + ';' — коректно відкривається в Excel
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)
    print(f"[write] {path}")


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        return list(csv.DictReader(f, delimiter=";" if sample.count(";") >= sample.count(",") else ","))


def cmd_review(_):
    gold, units = load_gold(), all_units()
    rows = []
    for g in gold:
        if not g.get("relevant"):
            rows.append([g["qid"], g["qtype"], g["query"], "—", "", "", "(без відповіді)", ""])
        for u, grade in g.get("relevant", {}).items():
            t, s, text = units.get(u, ("?", "?", "НЕ ЗНАЙДЕНО"))
            rows.append([g["qid"], g["qtype"], g["query"], u, grade, f"{t} › {s}", text[:600], ""])
    _write_csv(EVAL / "gold_review.csv",
               ["qid", "qtype", "query", "unit", "grade", "where", "text", "verdict (ok / remove / змінити)"], rows)


def cmd_pool(args):
    from hybrid import HybridSearcher
    gold = [g for g in load_gold() if g.get("relevant")][: args.n]
    hs = HybridSearcher(args.strategy, args.model)
    rows = []
    for g in gold:
        pool = {}
        for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
            ctxs, _ = hs.search(g["query"], mode, k=5)
            for c in ctxs:
                pool.setdefault(c["chunk_id"], c)
        for cid, c in sorted(pool.items()):          # сортування за id приховує, з якої системи фрагмент
            rows.append([g["qid"], g["query"], cid, " | ".join(c["units"]), f"{c['title']} › {c['section']}",
                         c["text"][:700].replace("\n", " "), "", ""])
    _write_csv(EVAL / "annotation_pool.csv",
               ["qid", "query", "chunk_id", "units", "where", "text", "label_A", "label_B"], rows)
    print(f"Пул: {len(gold)} запитів, {len(rows)} пар. Заповніть label_A (ви) і label_B (другий розмітник): 1 — релевантний, 0 — ні.")


def interpret(k):
    for thr, name in [(0.8, "майже повна згода"), (0.6, "значна згода"), (0.4, "помірна згода"),
                      (0.2, "слабка згода"), (0.0, "незначна згода")]:
        if k > thr:
            return name
    return "згода не краща за випадкову"


def cmd_autolabel(args):
    """Автоматичний розмітник: cross-encoder оцінює кожну пару (запит, фрагмент); 1, якщо score > поріг."""
    from sentence_transformers import CrossEncoder
    from common import RERANKER
    path = Path(args.file)
    rows = _read_csv(path)
    ce = CrossEncoder(RERANKER, max_length=512)
    scores = ce.predict([(r["query"], r["where"] + ". " + r["text"]) for r in rows], batch_size=16)
    for r, s in zip(rows, scores):
        r["label_A"] = "1" if float(s) > args.threshold else "0"
        r["ce_score"] = f"{float(s):.3f}"
    header = list(rows[0].keys())
    _write_csv(path, header, [[r[h] for h in header] for r in rows])
    print(f"label_A (cross-encoder, поріг {args.threshold}): релевантних {sum(r['label_A'] == '1' for r in rows)} з {len(rows)}")


def cmd_kappa(args):
    from metrics import cohen_kappa
    from common import RERANKER as RERANKER_NAME
    rows = [r for r in _read_csv(args.file) if str(r.get("label_A", "")).strip() and str(r.get("label_B", "")).strip()]
    if not rows:
        raise SystemExit("Немає рядків із заповненими label_A і label_B")
    a = [int(float(r["label_A"])) for r in rows]
    b = [int(float(r["label_B"])) for r in rows]
    k, po, pe = cohen_kappa(a, b)
    nq = len({r["qid"] for r in rows})
    conf = Counter(zip(a, b))
    res = {"kappa": k, "po": po, "pe": pe, "n_pairs": len(rows), "n_queries": nq, "interpretation": interpret(k), "annotators": f"A — cross-encoder {RERANKER_NAME} (автоматичний), B — LLM Claude Opus 5.5",
           "confusion": {f"A{x}_B{y}": n for (x, y), n in conf.items()}}
    (EVAL / "kappa.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    dis = [[r["qid"], r["chunk_id"], r["label_A"], r["label_B"]] for r in rows if r["label_A"] != r["label_B"]]
    write_text(EVAL / "kappa.md", f"""# Узгодженість розмітників (κ Коена)

Розмітник A — автоматичний (cross-encoder `{RERANKER_NAME}`, score > 0 → 1); розмітник B — LLM (Claude Opus 5.5).\nЗапитів: {nq}; пар (запит, фрагмент): {len(rows)}.

{md_table(["", "B = 1", "B = 0"], [["A = 1", conf[(1, 1)], conf[(1, 0)]], ["A = 0", conf[(0, 1)], conf[(0, 0)]]])}

p_o (спостережувана згода) = {po:.3f}; p_e (випадкова згода) = {pe:.3f};
**κ = (p_o − p_e) / (1 − p_e) = {k:.3f}** — {interpret(k)} (шкала Landis & Koch, 1977).

## Розбіжності ({len(dis)})
{md_table(["qid", "chunk_id", "A", "B"], dis) if dis else "немає"}
""")
    print(f"κ = {k:.3f} ({interpret(k)}), p_o={po:.3f}, p_e={pe:.3f}, пар={len(rows)}, запитів={nq}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("review")
    p = sub.add_parser("pool")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--strategy", default="structure")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p = sub.add_parser("autolabel")
    p.add_argument("--file", default=str(EVAL / "annotation_pool.csv"))
    p.add_argument("--threshold", type=float, default=0.0)
    p = sub.add_parser("kappa")
    p.add_argument("--file", default=str(EVAL / "annotation_pool.csv"))
    args = ap.parse_args()
    {"check": cmd_check, "review": cmd_review, "pool": cmd_pool, "autolabel": cmd_autolabel, "kappa": cmd_kappa}[args.cmd](args)


if __name__ == "__main__":
    main()
