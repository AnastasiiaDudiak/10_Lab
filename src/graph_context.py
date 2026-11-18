"""Результат 2. Підграф із графа знань (ЛР 9, EM-DAT 2023) для контексту RAG.

  1) entity linking: слова запиту -> вузли графа (Country, Region, Group, Subgroup, Type, Subtype)
     за назвою (EN) та українськими синонімами (SYNONYMS), з «легким стемінгом» (перші 5 символів);
  2) витяг підграфа: ієрархія типу (1–2 кроки BELONGS_TO) + події, що відповідають знайденим вузлам
     (1 крок: Event-[:OF_TYPE|IN_COUNTRY|IN_REGION|…]->вузол), з лімітом подій;
  3) лінеаризація з ID вузлів, які LLM може цитувати: [Type:Flood], [Event:2023-0054-TUR], [AGG:…].

Бекенд: Neo4j (якщо задано NEO4J_URI) або локальні CSV з ЛР 9 (graph/*.csv) — однакова логіка.
Перевірка:  python src/graph_context.py "скільки повеней було в Індії у 2023"
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, ROOT  # noqa: E402

GRAPH_DIR = ROOT / "graph"
LINK_KINDS = ["Country", "Region", "Group", "Subgroup", "Type", "Subtype"]
FILTER_REL = {"Country": "IN_COUNTRY", "Region": "IN_REGION", "Group": "IN_GROUP", "Subgroup": "IN_SUBGROUP",
              "Type": "OF_TYPE", "Subtype": "OF_SUBTYPE"}

# Українські синоніми -> id вузла (перевірено вручну за списком вузлів ЛР 9)
SYNONYMS = {
    "повін": "Type:Flood", "повен": "Type:Flood", "паводк": "Type:Flood", "затопл": "Type:Flood",
    "землет": "Type:Earthquake", "сейсм": "Type:Earthquake",
    "шторм": "Type:Storm", "буря": "Type:Storm", "ураган": "Subtype:Tropical cyclone",
    "циклон": "Subtype:Tropical cyclone", "тайфун": "Subtype:Tropical cyclone", "торнад": "Subtype:Tornado",
    "смерч": "Subtype:Tornado", "хуртов": "Subtype:Blizzard/Winter storm",
    "посух": "Type:Drought", "пожеж": "Type:Wildfire", "лісов": "Type:Wildfire",
    "епідем": "Type:Epidemic", "спалах": "Type:Epidemic", "холер": "Subtype:Bacterial disease",
    "спека": "Subtype:Heat wave", "спеки": "Subtype:Heat wave", "холод": "Subtype:Cold wave",
    "зсув": "Type:Mass movement (wet)", "вулка": "Type:Volcanic activity", "вивер": "Type:Volcanic activity",
    "лавин": "Subtype:Avalanche (wet)", "дорожн": "Type:Road", "автомоб": "Type:Road", "залізн": "Type:Rail",
    "авіак": "Type:Air", "авіац": "Type:Air", "вибух": "Type:Explosion (Industrial)",
    "природ": "Group:Natural", "техног": "Group:Technological",
    "гідрол": "Subgroup:Hydrological", "метеор": "Subgroup:Meteorological", "кліма": "Subgroup:Climatological",
    "геофіз": "Subgroup:Geophysical", "біолог": "Subgroup:Biological", "трансп": "Subgroup:Transport",
    "промисл": "Subgroup:Industrial accident",
    "азії": "Region:Asia", "азія": "Region:Asia", "афри": "Region:Africa", "європ": "Region:Europe",
    "амери": "Region:Americas", "океані": "Region:Oceania",
    "україн": "Country:Ukraine", "туреч": "Country:Türkiye", "лівії": "Country:Libya", "лівія": "Country:Libya",
    "малаві": "Country:Malawi", "індії": "Country:India", "індія": "Country:India", "китай": "Country:China",
    "китаї": "Country:China", "сша": "Country:United States of America", "япон": "Country:Japan",
    "мароко": "Country:Morocco", "марокк": "Country:Morocco", "мексик": "Country:Mexico", "пакист": "Country:Pakistan",
    "філіпп": "Country:Philippines", "індонез": "Country:Indonesia", "бразил": "Country:Brazil",
    "мозамб": "Country:Mozambique", "сирії": "Country:Syrian Arab Republic", "греці": "Country:Greece",
    "італі": "Country:Italy", "перу": "Country:Peru", "конго": "Country:Democratic Republic of the Congo",
}
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ'’()\-/]+")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- бекенди
class LocalGraph:
    """Граф ЛР 9 із CSV у пам'яті (fallback, коли Neo4j недоступний)."""
    backend = "local-csv (graph/*.csv, ЛР 9)"

    def __init__(self, gdir=GRAPH_DIR):
        nodes = list(csv.DictReader(open(gdir / "nodes.csv", encoding="utf-8-sig")))
        self.nodes = {n["id"]: {"id": n["id"], "kind": n["kind"], "name": n["name"]} for n in nodes}
        self.out = defaultdict(list)
        for r in csv.DictReader(open(gdir / "relationships.csv", encoding="utf-8-sig")):
            self.out[r["from"]].append((r["type"], r["to"]))
        for e in csv.DictReader(open(gdir / "events_2023.csv", encoding="utf-8-sig")):
            n = self.nodes.get("Event:" + e["id"])
            if n:
                n.update(deaths=_num(e["deaths"]), affected=_num(e["affected"]), country=e["country"],
                         region=e["region"], type=e["type"])
        _enrich_events(self.nodes)

    def vocab(self):
        return [n for n in self.nodes.values() if n["kind"] in LINK_KINDS]

    def parents(self, node_id):
        return [(t, self.nodes[b]) for t, b in self.out.get(node_id, []) if t == "BELONGS_TO" and b in self.nodes]

    def children(self, node_id):
        return [self.nodes[a] for a, rels in self.out.items() for t, b in rels if t == "BELONGS_TO" and b == node_id]

    def events(self, filters):
        """filters: {kind: [node_id,…]} — AND між видами, OR усередині виду."""
        res = []
        for nid, n in self.nodes.items():
            if n["kind"] != "Event":
                continue
            targets = {(t, b) for t, b in self.out.get(nid, [])}
            if all(any((FILTER_REL[k], v) in targets for v in vals) for k, vals in filters.items()):
                res.append(dict(n, subtype=next((b.split(":", 1)[1] for t, b in targets if t == "OF_SUBTYPE"), ""),
                                subgroup=next((b.split(":", 1)[1] for t, b in targets if t == "IN_SUBGROUP"), "")))
        return res


class Neo4jGraph:
    """Той самий інтерфейс поверх Neo4j (дані імпортує src/load_graph.py)."""
    backend = "Neo4j"

    def __init__(self, uri, user, password):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.backend = f"Neo4j ({uri})"

    def _q(self, cypher, **params):
        with self.driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

    def vocab(self):
        return self._q("MATCH (n:Entity) WHERE n.kind IN $kinds RETURN n.id AS id, n.kind AS kind, n.name AS name",
                       kinds=LINK_KINDS)

    def parents(self, node_id):
        rows = self._q("MATCH (a:Entity {id:$id})-[:BELONGS_TO]->(b) RETURN b.id AS id, b.kind AS kind, b.name AS name",
                       id=node_id)
        return [("BELONGS_TO", r) for r in rows]

    def children(self, node_id):
        return self._q("MATCH (a)-[:BELONGS_TO]->(b:Entity {id:$id}) RETURN a.id AS id, a.kind AS kind, a.name AS name",
                       id=node_id)

    def events(self, filters):
        where, params = [], {}
        for i, (kind, vals) in enumerate(filters.items()):
            where.append(f"EXISTS {{ MATCH (e)-[:{FILTER_REL[kind]}]->(x{i}) WHERE x{i}.id IN $v{i} }}")
            params[f"v{i}"] = vals
        cypher = ("MATCH (e:Event) " + ("WHERE " + " AND ".join(where) if where else "") +
                  " OPTIONAL MATCH (e)-[:OF_SUBTYPE]->(st) OPTIONAL MATCH (e)-[:IN_SUBGROUP]->(sg) "
                  "RETURN e.id AS id, 'Event' AS kind, e.name AS name, e.deaths AS deaths, e.affected AS affected, "
                  "e.country AS country, e.region AS region, e.disasterType AS type, e.eventName AS event_name, "
                  "e.startDate AS start, st.name AS subtype, sg.name AS subgroup")
        return self._q(cypher, **params)


def _enrich_events(nodes):
    """Назви подій і дати початку з вихідного CSV EM-DAT (якщо файл є) — для «Freddy», «Daniel» тощо."""
    path = next(iter(sorted(RAW.glob("*emdat*.csv"))), None)
    if not path:
        return
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    import io
    for r in csv.DictReader(io.StringIO(text)):
        n = nodes.get("Event:" + (r.get("DisNo.") or "").strip())
        if not n:
            continue
        name = (r.get("Event Name") or "").strip()
        if name:
            n["event_name"] = name
        y, m, d = (r.get("Start Year") or ""), (r.get("Start Month") or ""), (r.get("Start Day") or "")
        n["start"] = "-".join(x.split(".")[0].zfill(2) for x in (y, m, d) if x and x != "nan")


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        uri = os.getenv("NEO4J_URI")
        if uri:
            try:
                _GRAPH = Neo4jGraph(uri, os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
            except Exception as e:  # noqa: BLE001
                print(f"[graph] Neo4j недоступний ({str(e)[:80]}) — використовую локальні CSV")
        if _GRAPH is None:
            _GRAPH = LocalGraph()
    return _GRAPH


# ---------------------------------------------------------------- entity linking
def link_entities(query, graph=None):
    graph = graph or get_graph()
    q = query.lower()
    words = [w.strip("'’()-") for w in WORD_RE.findall(q)]
    found = {}
    for n in graph.vocab():                                  # точна назва (EN), напр. «Wildfire», «Türkiye»
        name = n["name"].lower()
        if len(name) >= 4 and re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", q):
            found[n["id"]] = n
    by_id = {n["id"]: n for n in graph.vocab()}
    for w in words:                                          # українські синоніми з «легким стемінгом»
        for stem, nid in SYNONYMS.items():
            if w.startswith(stem) and nid in by_id:
                found[nid] = by_id[nid]
    # якщо знайдено і Type, і його Subtype — лишаємо обидва; Group/Subgroup «природні» ігноруємо, якщо є конкретний Type
    return list(found.values())


# ---------------------------------------------------------------- підграф
def _fmt_event(e):
    bits = [e.get("type") or e.get("name", "")]
    if e.get("subtype"):
        bits[0] += f" ({e['subtype']})"
    if e.get("event_name"):
        bits.append(f"назва: {e['event_name']}")
    if e.get("subgroup"):
        bits.append(f"підгрупа: {e['subgroup']}")
    bits.append(f"країна: {e.get('country')}, регіон: {e.get('region')}")
    if e.get("start"):
        bits.append(f"початок: {e['start']}")
    if e.get("deaths") is not None:
        bits.append(f"загиблі: {int(e['deaths'])}")
    if e.get("affected") is not None:
        bits.append(f"постраждалі: {int(e['affected'])}")
    return f"[{e['id']}] " + "; ".join(bits)


def subgraph_context(query, graph=None, max_events=12, max_hops=2):
    """Повертає (текст підграфа, список ID вузлів, інформація про лінкінг)."""
    graph = graph or get_graph()
    ents = link_entities(query, graph)
    if not ents:
        return "", [], {"entities": []}
    lines, ids = [], []

    def add_id(x):
        if x not in ids:
            ids.append(x)

    # 1) ієрархія для таксономічних вузлів (до max_hops кроків угору) + дочірні вузли
    for n in ents:
        if n["kind"] in ("Type", "Subtype", "Subgroup", "Group"):
            chain, cur = [n], n
            for _ in range(max_hops if n["kind"] != "Subtype" else max_hops + 1):
                ps = graph.parents(cur["id"])
                if not ps:
                    break
                cur = ps[0][1]
                chain.append(cur)
            if len(chain) > 1:
                lines.append(" —BELONGS_TO→ ".join(f"[{c['id']}]" for c in chain))
                for c in chain:
                    add_id(c["id"])
            if n["kind"] in ("Subgroup", "Group", "Type"):
                ch = graph.children(n["id"])[:25]
                if ch:
                    lines.append(f"[{n['id']}] містить: " + ", ".join(f"[{c['id']}]" for c in ch))
                    for c in ch:
                        add_id(c["id"])
    # 2) події, що відповідають усім знайденим вузлам (AND між видами)
    filters = defaultdict(list)
    for n in ents:
        filters[n["kind"]].append(n["id"])
    evs = graph.events(dict(filters))
    if evs:
        agg_id = "AGG:" + "&".join(sorted(n["id"] for n in ents))
        deaths = sum(e["deaths"] or 0 for e in evs)
        countries = sorted({e.get("country") for e in evs if e.get("country")})
        line = (f"[{agg_id}] EM-DAT 2023, події, що відповідають {', '.join(n['id'] for n in ents)}: "
                f"кількість подій: {len(evs)}; загиблих разом: {int(deaths)}; країн: {len(countries)}")
        if len(countries) <= 20:
            line += " (" + ", ".join(countries) + ")"
        lines.append(line)
        add_id(agg_id)
        if "Type" not in filters and "Subtype" not in filters:
            dist = Counter(e.get("type") for e in evs).most_common(10)
            lines.append(f"[{agg_id}] розподіл за типами: " + ", ".join(f"{t}: {c}" for t, c in dist))
        elif "Subtype" not in filters:
            dist = Counter(e.get("subtype") for e in evs).most_common(10)
            lines.append(f"[{agg_id}] розподіл за підтипами: " + ", ".join(f"{t}: {c}" for t, c in dist))
        if "Country" not in filters and len(countries) > 1:
            cc = Counter(e.get("country") for e in evs).most_common(5)
            lines.append(f"[{agg_id}] країни з найбільшою кількістю подій: " + ", ".join(f"{c}: {k}" for c, k in cc))
        top = sorted(evs, key=lambda e: -(e["deaths"] or 0))[:max_events]
        lines.append(f"Події (до {max_events}, за кількістю загиблих):")
        for e in top:
            lines.append(_fmt_event(e))
            add_id(e["id"])
    info = {"entities": [n["id"] for n in ents], "n_events": len(evs) if evs else 0}
    return "\n".join(lines), ids, info


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "скільки повеней було в Індії у 2023 році"
    g = get_graph()
    txt, ids, info = subgraph_context(q, g)
    print(f"Бекенд: {g.backend}\nЗапит: {q}\nEntity linking: {info['entities']}\n---\n{txt}\n---\nID у контексті: {len(ids)}")
