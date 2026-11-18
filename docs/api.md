# REST API (Результат 3)

**Стек:** FastAPI + Pydantic, uvicorn. **Опис OpenAPI:** `docs/openapi.json`, інтерактивно — `http://127.0.0.1:8000/docs`.

| Ендпоінт | Опис | Авторизація |
|---|---|---|
| `GET /health` | стан сервісу, модель, розмір індексу, бекенд графа | ні |
| `POST /ask` | `{question, top_k, use_graph}` → відповідь, цитати, джерела, токени, вартість, затримка | `X-API-Key` + ліміт |
| `GET /sources/{id}` | повний текст джерела, якщо роль має доступ до його sensitivity | `X-API-Key` |

**Ролі:** `public` → лише `sensitivity=public` (Вікіпедія), без графа; `analyst` → усі рівні, включно з `restricted` (EM-DAT) і підграфом.
**Валідація:** `question` 3–500 символів, `top_k` 1–10, `id` за шаблоном `^[\w:.\-]+$` → інакше 422.
**Коди помилок:** 401 — немає/невірний ключ; 403 — недостатньо прав; 404 — джерело не знайдено; 422 — валідація;
429 — понад 3 запитів `/ask` за 60 с на ключ (заголовок `Retry-After`); 503 — LLM недоступна.

## Таблиця 3.1. Демонстрація (FastAPI TestClient)
| Перевірка | Запит | Роль | Код | Очікується | Результат | Відповідь (скорочено) |
|---|---|---|---|---|---|---|
| Стан сервісу | GET /health | — | 200 | 200 | PASS | {"status": "ok", "generator": "openai/gpt-oss-120b", "index_chunks": 2166, "graph": "local-csv (graph/*.csv, ЛР 9)", "uptime_s": 0.0} |
| Питання з EM-DAT, роль public | POST /ask | public | 200 | 200 | PASS | {"answer": "Не знаю: у наданих джерелах немає відповіді на це питання.", "abstained": true, "citations": [], "invalid_citations": [], "sources": [{"id": "wiki_uk_20:p0038", "title": "Міжнародна реакція на землетрус у Тур |
| Те саме питання, роль analyst | POST /ask | analyst | 200 | 200 | PASS | {"answer": "За 2023 рік у EM‑DAT зафіксовано три землетруси в Туреччині, загальна кількість загиблих — 50790 осіб [AGG:Country:Türkiye&Type:Earthquake][Event:2023-0054-TUR][Event:2023-0091-TUR][Event:2023-0107-TUR].", "a |
| Джерело restricted, роль public | GET /sources/emdat:2023-0054-TUR | public | 403 | 403 | PASS | {"detail": "Роль public не має доступу до джерел рівня restricted"} |
| Джерело restricted, роль analyst | GET /sources/emdat:2023-0054-TUR | analyst | 200 | 200 | PASS | {"id": "emdat:2023-0054-TUR", "title": "EM-DAT 2023: Earthquake", "section": "2023-0054-TUR", "sensitivity": "restricted", "source_url": "https://www.emdat.be/ (public_emdat_project.csv)", "date": "2023-02-06", "text": " |
| Неіснуюче джерело | GET /sources/wiki_uk_99:p9999 | analyst | 404 | 404 | PASS | {"detail": "Джерело wiki_uk_99:p9999 не знайдено"} |
| Без API-ключа | POST /ask | — | 401 | 401 | PASS | {"detail": "Потрібен заголовок X-API-Key"} |
| Невірний API-ключ | POST /ask | — | 401 | 401 | PASS | {"detail": "Невірний API-ключ"} |
| Валідація: питання 1 символ | POST /ask | analyst | 422 | 422 | PASS | {"detail": [{"type": "string_too_short", "loc": ["body", "question"], "msg": "String should have at least 3 characters", "input": "?", "ctx": {"min_length": 3}}]} |
| Валідація: top_k = 50 | POST /ask | analyst | 422 | 422 | PASS | {"detail": [{"type": "less_than_equal", "loc": ["body", "top_k"], "msg": "Input should be less than or equal to 10", "input": 50, "ctx": {"le": 10}}]} |
| Перевищення ліміту | POST /ask | analyst | 429 | 429 | PASS | {"detail": "Ліміт 3 запитів за 60 с; спробуйте через 58 с"} |

## Різні права для 2 ролей (одне й те саме питання)
**public:** Не знаю: у наданих джерелах немає відповіді на це питання.
Джерела: wiki_uk_20:p0038 (public), wiki_uk_20:p0048 (public), wiki_uk_20:p0054 (public), wiki_uk_20:p0031 (public), wiki_uk_20:p0000 (public)

**analyst:** За 2023 рік у EM‑DAT зафіксовано три землетруси в Туреччині, загальна кількість загиблих — 50790 осіб [AGG:Country:Türkiye&Type:Earthquake][Event:2023-0054-TUR][Event:2023-0091-TUR][Event:2023-0107-TUR].
Джерела: emdat:2023-0091-TUR (restricted), emdat:2023-0113-GRC (restricted), emdat:2023-0611-NGA (restricted), wiki_uk_20:p0038 (public), emdat:2023-0596-MMR (restricted); граф: Type:Earthquake, Subgroup:Geophysical, Group:Natural, Subtype:Ground movement, AGG:Country:Türkiye&Type:Earthquake

## Приклади curl (Windows PowerShell — `curl.exe`)
```
uvicorn api.main:app --port 8000
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/ask -H "X-API-Key: <ключ analyst>" -H "Content-Type: application/json" -d "{\"question\": \"що робити, якщо смерч застав на вулиці\", \"top_k\": 5}"
curl.exe http://127.0.0.1:8000/sources/emdat:2023-0054-TUR -H "X-API-Key: <ключ public>"
```
Ключі зберігаються лише в `.env` (не в коді й не в репозиторії).
