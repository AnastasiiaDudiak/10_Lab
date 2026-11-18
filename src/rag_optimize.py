"""Результат 5. Оптимізація «до/після»: контрольований експеримент з одним фактором — top-k контексту (5 → 3).

Гіпотеза: зменшення k з 5 до 3 скоротить вхідні токени (і вартість) приблизно на третину та знизить затримку,
а faithfulness і citation accuracy не погіршаться, бо reranker ставить найкорисніші джерела першими;
ризик — падіння context recall на багатокрокових і конфліктних питаннях.
Незмінні: корпус, стратегія chunking, моделі (embeddings, reranker, генератор, суддя), промпт, тести, seed.

  python src/rag_optimize.py          (потрібен прогін «до»: python src/rag_eval.py run)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOCS_DIR, EVAL, md_table, read_jsonl, write_text  # noqa: E402
import llm  # noqa: E402
from rag_eval import metrics, run_eval  # noqa: E402

BEFORE, AFTER, K_AFTER = "baseline_k5", "k3", 3


def main():
    before_path = EVAL / "runs" / f"rag_{BEFORE}.jsonl"
    if not before_path.exists():
        raise SystemExit("Немає прогону «до»: спершу python src/rag_eval.py run")
    before = read_jsonl(before_path)
    after = run_eval(AFTER, k=K_AFTER)
    mb, ma = metrics(before), metrics(after)

    def row(label, key, fmt="{:.3f}", better="up"):
        b, a = mb[key], ma[key]
        d = a - b
        sign = "↑" if d > 0 else "↓" if d < 0 else "="
        good = (d > 0 and better == "up") or (d < 0 and better == "down")
        verdict = "краще" if good else "гірше" if d != 0 else "без змін"
        return [label, fmt.format(b), fmt.format(a), f"{sign} {fmt.format(abs(d))}", verdict]

    q = md_table(["Метрика", "До (k = 5)", "Після (k = 3)", "Зміна", "Оцінка"], [
        row("Faithfulness", "faithfulness"), row("Context recall", "context_recall"),
        row("Citation accuracy", "citation_accuracy"), row("Abstention rate (без відповіді)", "abstention_correct"),
        row("Хибні відмови", "abstention_false", better="down"), row("Конфлікт визнано", "conflict_ack"),
        row("Injection відбито", "injection_resisted")])
    c = md_table(["Показник", "До (k = 5)", "Після (k = 3)", "Зміна"], [
        ["Затримка p50, мс", f"{mb['p50']:.0f}", f"{ma['p50']:.0f}", f"{(ma['p50'] - mb['p50']) / mb['p50'] * 100:+.0f} %"],
        ["Затримка p95, мс", f"{mb['p95']:.0f}", f"{ma['p95']:.0f}", f"{(ma['p95'] - mb['p95']) / mb['p95'] * 100:+.0f} %"],
        ["Вхідні токени / запит", f"{mb['prompt_tokens']:.0f}", f"{ma['prompt_tokens']:.0f}",
         f"{(ma['prompt_tokens'] - mb['prompt_tokens']) / mb['prompt_tokens'] * 100:+.0f} %"],
        ["Вихідні токени / запит", f"{mb['completion_tokens']:.0f}", f"{ma['completion_tokens']:.0f}",
         f"{(ma['completion_tokens'] - mb['completion_tokens']) / mb['completion_tokens'] * 100:+.0f} %"],
        ["Вартість / запит, $", f"{mb['cost']:.5f}", f"{ma['cost']:.5f}", f"{(ma['cost'] - mb['cost']) / mb['cost'] * 100:+.0f} %"],
        ["Вартість / 1000 запитів, $", f"{mb['cost'] * 1000:.2f}", f"{ma['cost'] * 1000:.2f}", ""],
    ])
    by = {r["id"]: r for r in before}
    changed = []
    for r in after:
        b = by[r["id"]]
        fb, fa = b["judge"]["faithfulness"], r["judge"]["faithfulness"]
        if b["abstained"] != r["abstained"] or (fb is not None and fa is not None and abs(fa - fb) >= 0.25) \
                or (b["context_recall"] is not None and r["context_recall"] is not None and b["context_recall"] != r["context_recall"]):
            changed.append([r["id"], r["category"], r["question"][:60],
                            f"{'не знаю' if b['abstained'] else fb}", f"{'не знаю' if r['abstained'] else fa}",
                            b["context_recall"], r["context_recall"]])
    ch = md_table(["id", "Кат.", "Питання", "Faith. до", "Faith. після", "Recall до", "Recall після"], changed) \
        if changed else "_Відповіді не змінились суттєво._"
    txt = f"""# Оптимізація «до/після» (Результат 5)

## 5.1. Гіпотеза і змінений фактор
**Фактор:** кількість текстових джерел у контексті **top-k: 5 → 3** (після reranking).
**Гіпотеза:** вхідні токени й вартість зменшаться приблизно на третину, затримка знизиться, а faithfulness і
citation accuracy не погіршаться, бо cross-encoder ставить найкорисніші фрагменти першими. Ризик — нижчий context recall
на багатокрокових і конфліктних питаннях (потрібні 2+ розділи).
**Контроль:** корпус, chunking (hierarchical), embeddings (MiniLM), reranker, генератор `{llm.GEN_MODEL}`, суддя `{llm.JUDGE_MODEL}`,
промпт `rag_v1`, 30 тестів, seed = 42 — без змін. Підграф додається в обох прогонах однаково.

## 5.2. Якість до/після
{q}

## 5.3. Затримка, токени, вартість
{c}

Затримка — пошук + граф + генерація; вартість — генератор за цінами Groq ({llm.PRICES.get(llm.GEN_MODEL)} $ за 1 млн вхідних/вихідних токенів).
На безкоштовному плані фактична вартість = 0 $; наведено вартість за тарифом Developer.

## 5.4. Тести, де результат змінився
{ch}

## 5.5. Висновок
_TODO: заповнюється після запуску за числами таблиць 5.2–5.3._
"""
    write_text(DOCS_DIR / "rag_optimization.md", txt)


if __name__ == "__main__":
    main()
