"""Результат 1. Збір і очищення корпусу, метадані, картка корпусу.

Джерела:
  1) Вікіпедія (uk + кілька en) через MediaWiki API — статті про НС (ліцензія CC BY-SA 4.0);
  2) kb/*.md з ЛР 2 (власні нотатки бази знань) — якщо папку вказано;
  3) EM-DAT 2023 (public_emdat_project.csv) — топ-N подій кожного типу як записи (restricted).

Запуск:  python src/build_corpus.py --kb-dir kb --emdat data/raw/public_emdat_project.csv
Результат: data/docs.jsonl, data/corpus.jsonl, docs/corpus_card.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DOCS_DIR, RAW, ROOT, md_table, write_jsonl, write_text  # noqa: E402

# (мова, назва статті / пошуковий запит). Назву, яку фактично знайдено, видно в картці корпусу.
TOPICS = [
    ("uk", "Надзвичайна ситуація"),
    ("uk", "Цивільний захист"),
    ("uk", "Державна служба України з надзвичайних ситуацій"),
    ("uk", "Стихійне лихо"),
    ("uk", "Землетрус"),
    ("uk", "Цунамі"),
    ("uk", "Повінь"),
    ("uk", "Тропічний циклон"),
    ("uk", "Торнадо"),
    ("uk", "Посуха"),
    ("uk", "Лісова пожежа"),
    ("uk", "Виверження вулкана"),
    ("uk", "Зсув"),
    ("uk", "Сель"),
    ("uk", "Снігова лавина"),
    ("uk", "Хвиля спеки"),
    ("uk", "Епідемія"),
    ("uk", "Пандемія COVID-19"),
    ("uk", "Чорнобильська катастрофа"),
    ("uk", "Руйнування греблі Каховської ГЕС"),
    ("uk", "Землетруси в Туреччині та Сирії 2023"),
    ("uk", "Землетрус в Індійському океані 2004"),
    ("uk", "Землетрус і цунамі в Тохоку 2011"),
    ("uk", "Ураган Катріна"),
    ("uk", "Евакуація"),
    ("uk", "Радіаційна аварія"),
    ("uk", "Медицина катастроф"),
    ("en", "Emergency management"),
    ("en", "Sendai Framework for Disaster Risk Reduction"),
    ("en", "Incident Command System"),
    ("en", "Early warning system"),
]

API = {"uk": "https://uk.wikipedia.org/w/api.php", "en": "https://en.wikipedia.org/w/api.php"}
UA = {"User-Agent": "KMS-Lab10-student/1.0 (educational project, Lviv University; https://www.lnu.edu.ua)"}

STOP_SECTIONS = {
    "див. також", "примітки", "джерела", "посилання", "література", "виноски", "галерея",
    "джерела та література", "посилання та джерела", "зовнішні посилання",
    "see also", "references", "external links", "further reading", "notes", "bibliography",
    "sources", "citations", "gallery", "footnotes",
}
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
NOISE_PATTERNS = [
    ("посилання-виноски [n]", re.compile(r"\[\d+\]")),
    ("службові позначки [джерело?]/[en]", re.compile(r"\[(?:[^\]\[]{1,25})\]")),
    ("залишки формул {\\displaystyle}", re.compile(r"\{\\displaystyle[^}]*\}")),
    ("стрілки виносок ↑", re.compile(r"↑")),
]


# ---------------------------------------------------------------- Wikipedia
def fetch_wiki(lang: str, term: str, sleep=2.0) -> dict | None:
    import requests
    slug = re.sub(r"\W+", "_", term.lower()).strip("_")
    cache = RAW / "wiki" / f"{lang}_{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    common = dict(action="query", format="json", formatversion=2, prop="extracts|info|revisions",
                  explaintext=1, exsectionformat="wiki", inprop="url", rvprop="timestamp", redirects=1)
    page = None
    for params in (dict(titles=term), dict(generator="search", gsrsearch=term, gsrlimit=1, gsrnamespace=0)):
        for attempt in range(6):            # повтор при 429 Too Many Requests / 5xx
            r = requests.get(API[lang], params={**common, **params}, headers=UA, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                wait = int(r.headers.get("Retry-After", 0) or 0) or 5 * (attempt + 1)
                print(f"  … сервер просить почекати ({r.status_code}), пауза {wait} с")
                time.sleep(wait)
                continue
            break
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", [])
        pages = [p for p in pages if not p.get("missing") and p.get("extract")]
        if pages:
            page = pages[0]
            break
        time.sleep(sleep)
    time.sleep(sleep)
    if page is None:
        print(f"  [!] не знайдено: {lang}:{term}")
        return None
    rec = {"lang": lang, "query": term, "title": page["title"], "url": page.get("fullurl"),
           "timestamp": (page.get("revisions") or [{}])[0].get("timestamp"), "extract": page["extract"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def clean_paragraphs(text: str, stats: Counter) -> list[str]:
    for name, rx in NOISE_PATTERNS:
        text, n = rx.subn("", text)
        stats[name] += n
    paras, seen = [], set()
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if len(line.split()) < 3:
            stats["короткі рядки (<3 слів: підписи, навігація)"] += 1
            continue
        if line in seen:
            stats["дублікати абзаців"] += 1
            continue
        seen.add(line)
        paras.append(line)
    return paras


def parse_wiki(rec: dict, stats: Counter) -> list[dict]:
    intro = "Вступ" if rec["lang"] == "uk" else "Introduction"
    sections, path, buf, skip_level = [], [intro], [], None
    current = intro

    def flush():
        if skip_level is None and buf:
            paras = clean_paragraphs("\n".join(buf), stats)
            if paras:
                sections.append({"section": current, "text": "\n".join(paras)})
            else:
                stats["порожні після очищення розділи"] += 1

    for line in rec["extract"].split("\n"):
        m = HEADING_RE.match(line.strip())
        if m:
            flush()
            buf = []
            level, name = len(m.group(1)), m.group(2).strip()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if name.lower() in STOP_SECTIONS:
                skip_level = level
                stats["службові розділи (Примітки, Посилання, Див. також…)"] += 1
            path = path[: level - 1] + [name]
            path = [p for p in path if p]
            current = " › ".join(path[1:]) if len(path) > 1 else path[0]
            continue
        buf.append(line)
    flush()
    return sections


# ---------------------------------------------------------------- kb/*.md (ЛР 2)
MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def parse_kb(path: Path, stats: Counter) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)          # YAML frontmatter
    text = MD_LINK.sub(r"\1", text)
    text = re.sub(r"[*_`>]+", "", text)
    title, sections, current, buf = path.stem, [], "Вступ", []

    def flush():
        paras = clean_paragraphs("\n".join(buf), stats)
        if paras:
            sections.append({"section": current, "text": "\n".join(paras)})

    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if m:
            if len(m.group(1)) == 1 and not sections and not buf:
                title = m.group(2).strip()
                continue
            flush()
            buf, current = [], m.group(2).strip()
        else:
            buf.append(line.lstrip("-•0123456789. ") if re.match(r"^\s*([-•]|\d+\.)\s", line) else line)
    flush()
    if not sections:
        return None
    date = dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    return {"title": title, "type": "kb_note", "lang": "uk", "date": date, "sensitivity": "internal",
            "source_url": f"kb/{path.name}", "license": "власна робота (ЛР 2)", "sections": sections,
            "atomic": False}


# ---------------------------------------------------------------- EM-DAT
def _get(row, *names):
    for n in names:
        for k in row:
            if k.strip().lower() == n.lower() and str(row[k]).strip() not in ("", "nan"):
                return str(row[k]).strip()
    return ""


def parse_emdat(path: Path, per_type: int, year: int = 2023) -> list[dict]:
    import io
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:            # Excel часто зберігає CSV у cp1252 (Türkiye, Valparaíso…)
        text = raw.decode("cp1252", errors="replace")
    sample = text[:4096]
    delim = ";" if sample.count(";") > sample.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    if year:                              # як і в попередніх ЛР — лише події заданого року
        def _year(r):
            y = _get(r, "Start Year") or _get(r, "DisNo.", "Dis No", "DisNo")[:4]
            return y.split(".")[0]
        rows = [r for r in rows if _year(r) == str(year)]
    by_type = defaultdict(list)
    for r in rows:
        t = _get(r, "Disaster Type")
        if t:
            by_type[t].append(r)

    def deaths(r):
        try:
            return float(_get(r, "Total Deaths") or 0)
        except ValueError:
            return 0.0

    docs = []
    for dtype, recs in sorted(by_type.items()):
        if len(recs) < 3:
            continue
        recs = sorted(recs, key=deaths, reverse=True)[:per_type]
        sections = []
        for r in recs:
            disno = _get(r, "DisNo.", "Dis No", "DisNo")
            y, m, d = _get(r, "Start Year"), _get(r, "Start Month"), _get(r, "Start Day")
            date = "-".join(x.split(".")[0].zfill(2) for x in (y, m, d) if x)
            parts = [f"Подія EM-DAT {disno}: {dtype}"]
            sub = _get(r, "Disaster Subtype")
            if sub:
                parts[0] += f" ({sub})"
            for label, val in [
                ("назва події", _get(r, "Event Name")), ("країна", _get(r, "Country")),
                ("регіон", _get(r, "Region")), ("місцевість", _get(r, "Location")),
                ("дата початку", date), ("загиблі", _get(r, "Total Deaths")),
                ("травмовані", _get(r, "No. Injured", "No Injured")),
                ("постраждалі загалом", _get(r, "Total Affected")),
                ("без житла", _get(r, "No. Homeless", "No Homeless")),
                ("збитки, тис. дол. США", _get(r, "Total Damage ('000 US$)", "Total Damage, Adjusted ('000 US$)")),
                ("магнітуда", (_get(r, "Magnitude") + " " + _get(r, "Magnitude Scale")).strip()),
                ("причина/походження", _get(r, "Origin")),
            ]:
                if val:
                    parts.append(f"{label}: {val}")
            sections.append({"section": disno or f"rec{len(sections)}", "text": "; ".join(parts) + ".",
                             "date": date})
        docs.append({"title": f"EM-DAT {year or ''}: {dtype}".replace("  ", " "), "type": "record", "lang": "uk",
                     "date": max((s["date"] for s in sections if s["date"]), default=""),
                     "sensitivity": "restricted", "source_url": "https://www.emdat.be/ (public_emdat_project.csv)",
                     "license": "EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення",
                     "sections": sections, "atomic": True})
    return docs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-dir", default="kb", help="папка з kb/*.md з ЛР 2 (якщо є)")
    ap.add_argument("--emdat", default=str(RAW / "public_emdat_project.csv"))
    ap.add_argument("--emdat-per-type", type=int, default=12)
    ap.add_argument("--emdat-year", type=int, default=2023, help="рік подій EM-DAT (0 — усі роки)")
    ap.add_argument("--min-words", type=int, default=150, help="мін. слів у документі після очищення")
    args = ap.parse_args()

    stats = Counter()
    docs = []
    print("== Вікіпедія ==")
    n_wiki = Counter()
    for lang, term in TOPICS:
        rec = fetch_wiki(lang, term)
        if not rec:
            continue
        sections = parse_wiki(rec, stats)
        words = sum(len(s["text"].split()) for s in sections)
        if words < args.min_words:
            print(f"  [-] {rec['title']}: лише {words} слів — пропущено")
            stats["документи < min_words (відкинуто)"] += 1
            continue
        n_wiki[lang] += 1
        docs.append({"doc_id": f"wiki_{lang}_{n_wiki[lang]:02d}", "title": rec["title"], "type": "article",
                     "lang": lang, "date": (rec["timestamp"] or "")[:10], "sensitivity": "public",
                     "source_url": rec["url"], "license": "CC BY-SA 4.0 (Вікіпедія)", "query": term,
                     "sections": sections, "atomic": False})
        print(f"  [+] {docs[-1]['doc_id']}: {rec['title']} — {len(sections)} розд., {words} слів")

    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_absolute():
        kb_dir = ROOT / kb_dir
    if kb_dir.exists():
        print("== kb/*.md (ЛР 2) ==")
        for i, p in enumerate(sorted(kb_dir.glob("*.md")), 1):
            d = parse_kb(p, stats)
            if d:
                d["doc_id"] = f"kb_{i:02d}"
                docs.append(d)
                print(f"  [+] {d['doc_id']}: {d['title']} — {len(d['sections'])} розд.")
    else:
        print(f"(папки {kb_dir} немає — kb/*.md пропущено)")

    em = Path(args.emdat)
    if not em.exists():                       # шукаємо будь-який CSV EM-DAT у data/ або data/raw/
        found = sorted(list(DATA.glob("*emdat*.csv")) + list(RAW.glob("*emdat*.csv")) + list(DATA.glob("*.csv")))
        if found:
            em = found[0]
    if em.exists():
        print(f"EM-DAT файл: {em}")
        print("== EM-DAT ==")
        for d in parse_emdat(em, args.emdat_per_type, args.emdat_year):
            slug = re.sub(r"\W+", "_", d["title"].split(":")[1].strip().lower())
            d["doc_id"] = f"emdat_{slug}"
            docs.append(d)
            print(f"  [+] {d['doc_id']}: {len(d['sections'])} записів")
    else:
        print(f"(файлу {em} немає — EM-DAT пропущено)")

    write_jsonl(DATA / "docs.jsonl", docs)

    from chunking import STRATEGIES, chunk_corpus  # noqa: E402
    sets = {name: chunk_corpus(docs, name) for name in STRATEGIES}
    corpus = sets["structure"].chunks
    write_jsonl(DATA / "corpus.jsonl", corpus)
    print(f"[write] data/corpus.jsonl — {len(corpus)} фрагментів (стратегія structure)")
    write_card(docs, sets, stats, args.emdat_per_type)

    ok_docs, ok_chunks = len(docs) >= 20, len(corpus) >= 200
    print(f"\nПЕРЕВІРКА C1: документів {len(docs)} ({'OK' if ok_docs else '<20!'}), "
          f"фрагментів {len(corpus)} ({'OK' if ok_chunks else '<200!'})")
    sens = Counter(c["sensitivity"] for c in corpus)
    print(f"ПЕРЕВІРКА C4: sensitivity задано для {sum(sens.values())}/{len(corpus)} фрагментів: {dict(sens)}")


def write_card(docs, sets, stats, per_type):
    corpus = sets["structure"].chunks
    per_doc = Counter(c["doc_id"] for c in corpus)
    langs = Counter(d["lang"] for d in docs)
    types = Counter(d["type"] for d in docs)
    sens_chunks = Counter(c["sensitivity"] for c in corpus)
    total_words = sum(len(s["text"].split()) for d in docs for s in d["sections"])
    rows = [[d["doc_id"], d["title"], d["type"], d["lang"], d["sensitivity"], d["date"], len(d["sections"]),
             sum(len(s["text"].split()) for s in d["sections"]), per_doc[d["doc_id"]], d["license"]]
            for d in docs]
    lic = Counter(d["license"] for d in docs)
    yrs = sorted({(sec.get("date") or "")[:4] for d in docs if d["type"] == "record" for sec in d["sections"]
                  if (sec.get("date") or "")[:4].isdigit()})
    em_years = ("події " + (yrs[0] if yrs[0] == yrs[-1] else f"{yrs[0]}–{yrs[-1]}") + " р.") if yrs else "не використано"
    kb_line = ("- **kb/*.md (ЛР 2)** — власні нотатки бази знань, `internal`." if types.get("kb_note")
               else "- Нотатки kb/*.md з ЛР 2 не використано (відсутні).")
    txt = f"""# Картка корпусу (Corpus Card) — ЛР 10

**Предметна область:** «Управління надзвичайними ситуаціями» (тема №30, проєкт Domain Knowledge Hub).
**Дата збирання:** {dt.date.today().isoformat()} · **Скрипт:** `src/build_corpus.py` (відтворюваний, сирі відповіді API кешуються в `data/raw/wiki/`).

## 1. Підсумок
| Показник | Значення |
|---|---|
| Документів | {len(docs)} |
| Фрагментів (стратегія `structure`, файл `data/corpus.jsonl`) | {len(corpus)} |
| Фрагментів за стратегіями | {", ".join(f"{k}: {len(v.chunks)}" for k, v in sets.items())} |
| Слів після очищення | {total_words} |
| Мови документів | {", ".join(f"{k}: {v}" for k, v in langs.items())} |
| Типи документів | {", ".join(f"{k}: {v}" for k, v in types.items())} |
| Фрагменти за sensitivity | {", ".join(f"{k}: {v}" for k, v in sens_chunks.items())} |

## 2. Джерела та ліцензії
{md_table(["Ліцензія / умови", "Документів"], lic.items())}

- **Вікіпедія (uk, en)** — тексти статей отримано через MediaWiki API (`prop=extracts`, plain text). Ліцензія CC BY-SA 4.0: дозволено використання з зазначенням авторства й посиланням (`source_url` у кожному фрагменті).
- **EM-DAT** ({em_years}; CRED / UCLouvain) — дані надано для некомерційного використання після реєстрації; повторне поширення сирих даних заборонене, тому фрагменти позначено `restricted` і не публікуються поза репозиторієм курсу.
{kb_line}

## 3. Метадані фрагмента
`chunk_id, doc_id, title, section, type, date, sensitivity, source_url, lang, text, n_words, units, strategy, context_id`
(`date` — дата останньої редакції статті / дата початку події EM-DAT / дата зміни файлу kb; `units` — розділи документа, які покриває фрагмент, використовуються для розмітки релевантності).

## 4. Правила очищення від шуму
1. Беремо лише текст статті (`explaintext`) — меню, навігаційні шаблони, інфобокси, категорії й колонтитули сторінки відкидаються API.
2. Видаляються службові розділи разом із підрозділами: «Примітки», «Джерела», «Посилання», «Література», «Див. також», «Галерея» (та en-відповідники).
3. Видаляються виноски `[12]`, позначки `[джерело?]`, `[en]`, залишки формул `{{\\displaystyle…}}`, стрілки `↑`.
4. Рядки коротші за 3 слова (підписи, залишки списків-навігації) відкидаються; пробіли нормалізуються.
5. Повторювані абзаци в межах документа видаляються.
6. Документи, коротші за 150 слів після очищення, відкидаються.

**Статистика застосування правил:**
{md_table(["Правило", "Спрацювань"], sorted(stats.items(), key=lambda x: -x[1]))}

## 5. Обмеження доступу (sensitivity)
| Рівень | Правило | Хто бачить |
|---|---|---|
| `public` | відкриті ліцензії (Вікіпедія) | усі |
| `internal` | власні нотатки команди (kb з ЛР 2) | учасники проєкту |
| `restricted` | дані з ліцензійними обмеженнями (EM-DAT) | лише авторизовані аналітики; виключаються фільтром для ролі `public` |

## 6. Документи
{md_table(["doc_id", "Назва", "Тип", "Мова", "Sensitivity", "Дата", "Розділів", "Слів", "Фрагм.", "Ліцензія"], rows)}

## 7. Відомі прогалини та упередження
- Вікіпедія — енциклопедичне джерело, а не нормативне: немає чинних регламентів ДСНС, планів реагування, наказів; деякі статті uk-Вікіпедії короткі або застарілі.
- З EM-DAT ({em_years}) взято лише топ-{per_type} подій кожного типу за кількістю загиблих — дрібні події не представлені.\n- Пошук Вікіпедії для кількох тем повернув вужчі статті («Хвилі спеки в Європі (2026)», «Міжнародна реакція на землетрус у Туреччині та Сирії 2023 року»); вони великі й переважають за кількістю фрагментів; стаття «Сель» відкинута як закоротка.
- Корпус переважно україномовний; англомовних документів мало (вони потрібні для перевірки багатомовності моделей).
- `date` для статей — дата останньої редакції, а не дата події/публікації.
- Числові дані у статтях можуть розходитися з EM-DAT (різні методики підрахунку).
"""
    write_text(DOCS_DIR / "corpus_card.md", txt)


if __name__ == "__main__":
    main()
