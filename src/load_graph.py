"""Імпорт графа ЛР 9 (graph/*.csv) у Neo4j через Python-драйвер — без копіювання файлів у папку import.

  python src/load_graph.py            (потрібні NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD у .env)
Працює з Neo4j Desktop (bolt://localhost:7687) і Neo4j AuraDB Free (neo4j+s://xxxx.databases.neo4j.io).
Ідемпотентний (MERGE): повторний запуск не створює дублікатів.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402,F401  (завантажує .env)
from graph_context import LocalGraph  # noqa: E402


def batched(rows, n=500):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def main():
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI")
    if not uri:
        raise SystemExit("Задайте NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD у файлі .env")
    g = LocalGraph()                       # читає CSV + назви/дати подій з EM-DAT CSV
    drv = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")))
    with drv.session() as s:
        for c in ["CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
                  "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)"]:
            s.run(c)
        nodes = list(g.nodes.values())
        for b in batched(nodes):
            s.run("UNWIND $rows AS r MERGE (n:Entity {id:r.id}) SET n.name=r.name, n.kind=r.kind, "
                  "n.deaths=r.deaths, n.affected=r.affected, n.country=r.country, n.region=r.region, "
                  "n.disasterType=r.type, n.eventName=r.event_name, n.startDate=r.start", rows=b)
        for kind in ["Event", "Country", "Region", "Group", "Subgroup", "Type", "Subtype"]:
            s.run(f"MATCH (n:Entity {{kind:'{kind}'}}) SET n:{kind}")
        rels = [{"a": a, "b": b, "t": t} for a, lst in g.out.items() for t, b in lst]
        for t in sorted({r["t"] for r in rels}):
            sub = [r for r in rels if r["t"] == t]
            for b in batched(sub):
                s.run(f"UNWIND $rows AS r MATCH (a:Entity {{id:r.a}}), (b:Entity {{id:r.b}}) "
                      f"MERGE (a)-[x:{t}]->(b) SET x.source='EM-DAT public CSV (ЛР 9)'", rows=b)
        n = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        m = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"Імпортовано: вузлів {n}, зв'язків {m} (очікується 779 / 4150)")


if __name__ == "__main__":
    main()
