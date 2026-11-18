"""Демонстрація API для звіту: 2 ролі, коди помилок, ліміт → docs/api.md, docs/openapi.json.

  python api/demo.py
(використовує FastAPI TestClient — сервер запускати не треба; ключі беруться з API_KEYS у .env)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RATE_LIMIT", "3")          # маленький ліміт, щоб показати 429
import llm  # noqa: E402,F401
os.environ["RATE_LIMIT"] = os.environ.get("DEMO_RATE_LIMIT", "3")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app, load_keys  # noqa: E402
from common import md_table, write_text  # noqa: E402


def main():
    keys = load_keys()
    pub = next((k for k, r in keys.items() if r == "public"), None)
    ana = next((k for k, r in keys.items() if r == "analyst"), None)
    if not pub or not ana:
        raise SystemExit("У .env потрібен API_KEYS з ключем public і ключем analyst, напр. API_KEYS=pub-123:public,ana-456:analyst")
    rows, examples = [], []
    q_restricted = "скільки людей загинуло під час землетрусу в Туреччині у 2023 році за даними EM-DAT"
    with TestClient(app) as c:
        def call(name, method, url, key=None, body=None, expect=None):
            h = {"X-API-Key": key} if key else {}
            r = c.request(method, url, headers=h, json=body)
            try:
                js = r.json()
            except Exception:  # noqa: BLE001
                js = r.text
            short = json.dumps(js, ensure_ascii=False)[:220]
            rows.append([name, f"{method} {url}", "public" if key == pub else "analyst" if key == ana else "—",
                         r.status_code, expect, "PASS" if r.status_code == expect else "FAIL", short.replace("|", "/")])
            print(f"{name}: {r.status_code} (очікується {expect})")
            return r, js

        call("Стан сервісу", "GET", "/health", None, None, 200)
        _, a_pub = call("Питання з EM-DAT, роль public", "POST", "/ask", pub, {"question": q_restricted}, 200)
        _, a_ana = call("Те саме питання, роль analyst", "POST", "/ask", ana, {"question": q_restricted}, 200)
        rid = "emdat:2023-0054-TUR"
        call("Джерело restricted, роль public", "GET", f"/sources/{rid}", pub, None, 403)
        call("Джерело restricted, роль analyst", "GET", f"/sources/{rid}", ana, None, 200)
        call("Неіснуюче джерело", "GET", "/sources/wiki_uk_99:p9999", ana, None, 404)
        call("Без API-ключа", "POST", "/ask", None, {"question": q_restricted}, 401)
        call("Невірний API-ключ", "POST", "/ask", "wrong-key", {"question": q_restricted}, 401)
        call("Валідація: питання 1 символ", "POST", "/ask", ana, {"question": "?"}, 422)
        call("Валідація: top_k = 50", "POST", "/ask", ana, {"question": "що таке цунамі", "top_k": 50}, 422)
        # ліміт: ключ analyst уже зробив 1 запит /ask; ще кілька однакових (відповідь LLM у кеші)
        for i in range(int(os.environ["RATE_LIMIT"])):
            r = c.post("/ask", headers={"X-API-Key": ana}, json={"question": q_restricted})
            if r.status_code == 429:
                break
        call("Перевищення ліміту", "POST", "/ask", ana, {"question": q_restricted}, 429)
        spec = c.get("/openapi.json").json()
    (ROOT / "docs" / "openapi.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    def src_list(js):
        return ", ".join(f"{s['id']} ({s['sensitivity']})" for s in js.get("sources", [])) if isinstance(js, dict) else ""
    txt = f"""# REST API (Результат 3)

**Стек:** FastAPI + Pydantic, uvicorn. **Опис OpenAPI:** `docs/openapi.json`, інтерактивно — `http://127.0.0.1:8000/docs`.

| Ендпоінт | Опис | Авторизація |
|---|---|---|
| `GET /health` | стан сервісу, модель, розмір індексу, бекенд графа | ні |
| `POST /ask` | `{{question, top_k, use_graph}}` → відповідь, цитати, джерела, токени, вартість, затримка | `X-API-Key` + ліміт |
| `GET /sources/{{id}}` | повний текст джерела, якщо роль має доступ до його sensitivity | `X-API-Key` |

**Ролі:** `public` → лише `sensitivity=public` (Вікіпедія), без графа; `analyst` → усі рівні, включно з `restricted` (EM-DAT) і підграфом.
**Валідація:** `question` 3–500 символів, `top_k` 1–10, `id` за шаблоном `^[\\w:.\\-]+$` → інакше 422.
**Коди помилок:** 401 — немає/невірний ключ; 403 — недостатньо прав; 404 — джерело не знайдено; 422 — валідація;
429 — понад {os.environ['RATE_LIMIT']} запитів `/ask` за 60 с на ключ (заголовок `Retry-After`); 503 — LLM недоступна.

## Таблиця 3.1. Демонстрація (FastAPI TestClient)
{md_table(["Перевірка", "Запит", "Роль", "Код", "Очікується", "Результат", "Відповідь (скорочено)"], rows)}

## Різні права для 2 ролей (одне й те саме питання)
**public:** {a_pub.get('answer', '') if isinstance(a_pub, dict) else ''}
Джерела: {src_list(a_pub)}

**analyst:** {a_ana.get('answer', '') if isinstance(a_ana, dict) else ''}
Джерела: {src_list(a_ana)}; граф: {', '.join(a_ana.get('graph_ids', [])[:5]) if isinstance(a_ana, dict) else ''}

## Приклади curl (Windows PowerShell — `curl.exe`)
```
uvicorn api.main:app --port 8000
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/ask -H "X-API-Key: <ключ analyst>" -H "Content-Type: application/json" -d "{{\\"question\\": \\"що робити, якщо смерч застав на вулиці\\", \\"top_k\\": 5}}"
curl.exe http://127.0.0.1:8000/sources/emdat:2023-0054-TUR -H "X-API-Key: <ключ public>"
```
Ключі зберігаються лише в `.env` (не в коді й не в репозиторії).
"""
    write_text(ROOT / "docs" / "api.md", txt)


if __name__ == "__main__":
    main()
