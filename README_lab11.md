# ЛР 11 · Чат-бот на основі гібридного сховища (RAG + граф)
Гілка `lab11`. Використовує індекс і золотий набір ЛР 10 (`src/hybrid.py`, `eval/gold.jsonl`) і граф ЛР 9 (`graph/*.csv`).

## Структура
```
src/llm.py            клієнт LLM (Groq, OpenAI-сумісний API): кеш, повтори при 429, токени, вартість
prompts/rag_v1.md     промпт: лише з контексту, цитати [id], «не знаю», конфлікт джерел, захист від injection
src/rag.py            Р1  конвеєр: роль → гібридний пошук → reranker → (+граф) → LLM → перевірка цитат → logs/rag_log.jsonl
src/graph_context.py  Р2  entity linking + підграф з лімітом + лінеаризація з ID (Neo4j або локальні CSV ЛР 9)
src/load_graph.py         імпорт графа ЛР 9 у Neo4j (необов'язково)
src/graph_compare.py  Р2  12 питань «лише текст» vs «текст + підграф» → docs/graph_vs_text.md
api/main.py           Р3  FastAPI: GET /health, POST /ask, GET /sources/{id}; X-API-Key, ролі, ліміт
api/demo.py           Р3  демонстрація 2 ролей і кодів помилок → docs/api.md, docs/openapi.json
eval/rag_eval.jsonl   Р4  30 тестів: 15 з відповіддю, 8 без, 4 конфлікти, 3 prompt injection
src/rag_eval.py       Р4  метрики + LLM-суддя + ручна перевірка (eval/judge_check.csv) → docs/rag_eval.md
src/rag_optimize.py   Р5  експеримент top-k 5 → 3 → docs/rag_optimization.md
```

## Встановлення
```
pip install -r requirements_lab11.txt
copy .env.example .env        (і вписати GROQ_API_KEY та API_KEYS)
```
Моделі: генератор `openai/gpt-oss-120b`, суддя `openai/gpt-oss-20b` (Groq, безкоштовний план: 8K токенів/хв, 200K/день на модель).
Відповіді LLM кешуються в `data/llm_cache/` — повторний запуск не витрачає ліміт. Seed = 42, temperature = 0.

## Запуск з нуля
```
python src/graph_context.py "скільки повеней було в Індії у 2023 році"
python src/rag.py --demo
python src/graph_compare.py
python api/demo.py
uvicorn api.main:app --port 8000          (http://127.0.0.1:8000/docs)
python src/rag_eval.py run
python src/rag_eval.py agree              (після заповнення eval/judge_check.csv)
python src/rag_optimize.py
```
Neo4j (необов'язково): задати NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD у .env, виконати `python src/load_graph.py`;
без цього граф читається з `graph/*.csv` тією самою логікою.
