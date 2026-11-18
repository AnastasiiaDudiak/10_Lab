"""Результат 2 (крок 3–4). Порівняння «лише текст» vs «текст + підграф» на 12 питаннях про зв'язки.

Еталонні відповіді (reference) обчислено з даних графа ЛР 9 (graph/events_2023.csv, relationships.csv),
тому їх можна перевірити запитом Cypher або в Excel.
  python src/graph_compare.py   →  docs/graph_vs_text.md, eval/runs/graph_compare.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOCS_DIR, EVAL, md_table, write_jsonl, write_text  # noqa: E402
import llm  # noqa: E402

QUESTIONS = [
    ("g01", "Скільки повеней зафіксовано в EM-DAT у 2023 році в Індії?", "5 подій типу Flood в Індії (India) у 2023 році."),
    ("g02", "До якої підгрупи і групи лих належать лісові пожежі (Wildfire)?", "Підгрупа Climatological, група Natural (природні)."),
    ("g03", "Які підтипи штормів були у 2023 році і який із них найчастіший?",
     "Найчастіший — Tropical cyclone (56 подій); також Severe weather 23, Storm (General) 21, Extra-tropical storm 12, Tornado 8, Blizzard/Winter storm 8, Sand/Dust storm 4, Lightning/Thunderstorms 4, Hail 2, Storm surge 1."),
    ("g04", "У яких країнах Африки у 2023 році сталися землетруси?", "Лише в Марокко (Morocco)."),
    ("g05", "Який землетрус 2023 року мав найбільше загиблих і в якій країні він стався?",
     "Подія 2023-0054-TUR у Туреччині (Türkiye), 50 783 загиблих."),
    ("g06", "Скільки техногенних лих було в Європі у 2023 році?", "11 подій групи Technological у регіоні Europe."),
    ("g07", "Які типи лих зафіксовано в Україні у 2023 році?", "Три типи, по одній події: Air (авіакатастрофа), Flood (повінь), Storm (шторм)."),
    ("g08", "Скільки людей загинуло від повеней в Африці у 2023 році і скільки було таких подій?",
     "38 повеней (Flood) в Африці, разом 4 296 загиблих."),
    ("g09", "Які типи лих належать до кліматологічної підгрупи?", "Drought (посуха), Glacial lake outburst flood, Wildfire (лісові пожежі)."),
    ("g10", "У скількох країнах Азії у 2023 році були тропічні циклони?",
     "12 країн: Bangladesh, China, Hong Kong (China), India, Japan, Myanmar, Pakistan, Philippines, Republic of Korea, Taiwan, Viet Nam, Yemen."),
    ("g11", "У якій країні у 2023 році було найбільше дорожньо-транспортних катастроф (Road)?", "Мексика (Mexico) — 6 подій; далі Peru — 5."),
    ("g12", "Які лиха 2023 року зафіксовано в Лівії і до яких підгруп вони належать?",
     "Дві події: шторм Daniel (2023-0610-LBY, Storm, підгрупа Meteorological, 12 352 загиблих) і аварія на воді (2023-0840-LBY, Water, підгрупа Transport, 60 загиблих)."),
]

JUDGE_CORRECT = """Ти — суворий екзаменатор. Порівняй ВІДПОВІДЬ системи з ЕТАЛОНОМ.
Оцінка correctness: 1 — відповідь містить усі ключові факти еталону (числа, назви) без суперечностей;
0.5 — частково правильна (частина фактів або неточні числа); 0 — неправильна, суперечить еталону або «не знаю».
Поверни лише JSON: {"correctness": 0|0.5|1, "explanation": "коротко українською"}

ПИТАННЯ: {q}
ЕТАЛОН: {ref}
ВІДПОВІДЬ: {a}"""


def judge_correct(q, ref, answer):
    rec = llm.chat([{"role": "user", "content": JUDGE_CORRECT.replace("{q}", q).replace("{ref}", ref).replace("{a}", answer)}],
                   model=llm.JUDGE_MODEL, max_tokens=300, json_mode=True)
    js = llm.parse_json(rec["text"]) or {}
    try:
        sc = float(js.get("correctness", 0))
    except (TypeError, ValueError):
        sc = 0.0
    return sc, js.get("explanation", ""), rec


def main():
    from rag import RagPipeline
    rag = RagPipeline(use_graph=True)
    rows, runs = [], []
    for qid, q, ref in QUESTIONS:
        out = {}
        for mode, ug in (("text", False), ("text+graph", True)):
            r = rag.ask(q, role="analyst", use_graph=ug, tag=f"graph_compare:{mode}")
            sc, why, _ = judge_correct(q, ref, r["answer"])
            graph_cites = [c for c in r["citations"] if c in set(r["graph_ids"])]
            out[mode] = dict(correct=sc, why=why, prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
                             latency=r["latency_ms"]["total"], cites=len(r["citations"]), valid=r["citations_valid"],
                             graph_cites=len(graph_cites), explainable=int(bool(r["citations"]) and r["citations_valid"]),
                             answer=r["answer"], entities=r["graph_entities"], cost=r["cost_usd"])
            print(f"{qid} {mode:11s} correct={sc} tokens={r['prompt_tokens']} cites={len(r['citations'])}")
        runs.append({"qid": qid, "question": q, "reference": ref, **{m: v for m, v in out.items()}})
        t, g = out["text"], out["text+graph"]
        rows.append([qid, q, ", ".join(g["entities"]) or "—", t["correct"], g["correct"], t["explainable"], g["explainable"],
                     g["graph_cites"], t["prompt_tokens"], g["prompt_tokens"]])
    write_jsonl(EVAL / "runs" / "graph_compare.jsonl", runs)

    def avg(mode, key):
        return float(np.mean([r[mode][key] for r in runs]))
    tc, gc = avg("text", "correct"), avg("text+graph", "correct")
    te, ge = avg("text", "explainable"), avg("text+graph", "explainable")
    tt, gt = avg("text", "prompt_tokens"), avg("text+graph", "prompt_tokens")
    tcost, gcost = avg("text", "cost"), avg("text+graph", "cost")
    tl, gl = avg("text", "latency"), avg("text+graph", "latency")
    summary = md_table(["Метрика", "Лише текст", "Текст + підграф", "Різниця"], [
        ["Правильність (суддя, 0–1), середня", f"{tc:.2f}", f"{gc:.2f}", f"{gc - tc:+.2f}"],
        ["Повністю правильних (=1)", sum(r['text']['correct'] == 1 for r in runs), sum(r['text+graph']['correct'] == 1 for r in runs), ""],
        ["Пояснюваність: відповідь з валідними цитатами, частка", f"{te:.2f}", f"{ge:.2f}", f"{ge - te:+.2f}"],
        ["Цитат на вузли графа, середньо", "0", f"{avg('text+graph', 'graph_cites'):.1f}", ""],
        ["Вхідні токени, середньо", f"{tt:.0f}", f"{gt:.0f}", f"{(gt - tt) / tt * 100:+.0f} %"],
        ["Вартість / запит, $", f"{tcost:.5f}", f"{gcost:.5f}", f"{gcost - tcost:+.5f}"],
        ["Затримка, мс, середня", f"{tl:.0f}", f"{gl:.0f}", f"{gl - tl:+.0f}"],
    ])
    detail = md_table(["qid", "Питання", "Entity linking", "Прав. текст", "Прав. +граф", "Пояс. текст", "Пояс. +граф",
                       "Цитат графа", "Токени текст", "Токени +граф"], rows)
    ex = runs[0]
    txt = f"""# Підграф vs лише текст (Результат 2)

**Граф:** EM-DAT 2023 з ЛР 9 (779 вузлів, 4150 зв'язків; бекенд: `{rag.graph.backend}`).
**Entity linking:** назви вузлів (EN) + українські синоніми зі стемінгом (`SYNONYMS` у `src/graph_context.py`).
**Підграф:** ієрархія BELONGS_TO (до 2 кроків) + події, пов'язані зі знайденими вузлами (1 крок), агрегати;
ліміт — 12 подій за кількістю загиблих. Лінеаризовано з ID вузлів (`[Type:Flood]`, `[Event:2023-0054-TUR]`, `[AGG:…]`).
**Правильність** оцінює LLM-суддя `{llm.JUDGE_MODEL}` за еталоном, обчисленим із даних графа (1 / 0.5 / 0).
**Пояснюваність** — відповідь має ≥ 1 цитату і всі цитати валідні.

## Таблиця 2.1. Підсумок ({len(runs)} питань)
{summary}

## Таблиця 2.2. За питаннями
{detail}

## Приклад: {ex['qid']} — {ex['question']}
**Еталон:** {ex['reference']}

**Лише текст:** {ex['text']['answer']}

**Текст + підграф:** {ex['text+graph']['answer']}

## Висновок
_TODO: заповнюється після запуску за числами таблиці 2.1._
"""
    write_text(DOCS_DIR / "graph_vs_text.md", txt)


if __name__ == "__main__":
    main()
