"""Результат 3. REST API чат-бота (FastAPI).

  uvicorn api.main:app --port 8000        → http://127.0.0.1:8000/docs (Swagger / OpenAPI)

Автентифікація: заголовок X-API-Key. Ключі та ролі — у .env: API_KEYS="ключ1:public,ключ2:analyst"
Ролі: public — лише джерела sensitivity=public; analyst — public + internal + restricted (EM-DAT, граф).
Ліміт: RATE_LIMIT запитів /ask на ключ за 60 с (за замовчуванням 10) → 429 + Retry-After.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Path as PathParam, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402  (завантажує .env)

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))
RATE_WINDOW = 60.0
LEVEL = {"public": 0, "internal": 1, "restricted": 2}
ROLE_MAX = {"public": "public", "analyst": "restricted"}


def load_keys():
    keys = {}
    for pair in os.getenv("API_KEYS", "").split(","):
        if ":" in pair:
            k, role = pair.strip().split(":", 1)
            if role in ROLE_MAX and k:
                keys[k] = role
    return keys


STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag import RagPipeline
    STATE["rag"] = RagPipeline()
    STATE["keys"] = load_keys()
    STATE["hits"] = defaultdict(deque)
    STATE["started"] = time.time()
    yield


app = FastAPI(title="Domain Knowledge Hub — НС чат-бот (ЛР 11)", version="1.0.0", lifespan=lifespan,
              description="RAG + граф знань: відповіді з цитатами [id] і політикою «не знаю». "
                          "Автентифікація — заголовок `X-API-Key`.")


# ---------------- моделі
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="Питання українською або англійською")
    top_k: int = Field(5, ge=1, le=10, description="Кількість текстових джерел у контексті")
    use_graph: bool = Field(True, description="Додавати підграф із графа знань (лише для ролі analyst)")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("питання не може бути порожнім")
        return v.strip()


class Source(BaseModel):
    id: str
    title: str
    section: str
    sensitivity: str
    score: float


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[str]
    invalid_citations: list[str]
    sources: list[Source]
    graph_ids: list[str]
    role: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


class SourceResponse(BaseModel):
    id: str
    title: str
    section: str
    sensitivity: str
    source_url: str
    date: str | None = None
    text: str


class ErrorResponse(BaseModel):
    detail: str


ERRORS = {401: {"model": ErrorResponse, "description": "Немає або невірний API-ключ"},
          403: {"model": ErrorResponse, "description": "Недостатньо прав для джерела"},
          404: {"model": ErrorResponse, "description": "Не знайдено"},
          429: {"model": ErrorResponse, "description": "Перевищено ліміт запитів"},
          503: {"model": ErrorResponse, "description": "LLM тимчасово недоступна"}}


# ---------------- залежності
def auth(x_api_key: str | None = Header(None, alias="X-API-Key")) -> dict:
    if not x_api_key:
        raise HTTPException(401, "Потрібен заголовок X-API-Key")
    role = STATE["keys"].get(x_api_key)
    if not role:
        raise HTTPException(401, "Невірний API-ключ")
    return {"key": x_api_key, "role": role}


def rate_limit(user: dict = Depends(auth)) -> dict:
    now, q = time.time(), STATE["hits"][user["key"]]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        retry = int(RATE_WINDOW - (now - q[0])) + 1
        raise HTTPException(429, f"Ліміт {RATE_LIMIT} запитів за {int(RATE_WINDOW)} с; спробуйте через {retry} с",
                            headers={"Retry-After": str(retry)})
    q.append(now)
    return user


@app.exception_handler(llm.LLMError)
async def llm_error(_: Request, exc: llm.LLMError):
    return JSONResponse(status_code=503, content={"detail": f"LLM недоступна: {str(exc)[:200]}"})


@app.exception_handler(llm.DailyLimitError)
async def llm_daily(_: Request, exc: llm.DailyLimitError):
    return JSONResponse(status_code=503, content={"detail": "Вичерпано денний ліміт LLM"}, headers={"Retry-After": "3600"})


# ---------------- ендпоінти
@app.get("/health", summary="Стан сервісу")
def health():
    rag = STATE.get("rag")
    return {"status": "ok" if rag else "starting", "generator": llm.GEN_MODEL,
            "index_chunks": len(rag.searcher.chunks) if rag else 0,
            "graph": rag.graph.backend if rag and rag.graph else None,
            "uptime_s": round(time.time() - STATE.get("started", time.time()), 1)}


@app.post("/ask", response_model=AskResponse, responses=ERRORS, summary="Питання → відповідь із джерелами")
def ask(req: AskRequest, user: dict = Depends(rate_limit)):
    r = STATE["rag"].ask(req.question, role=user["role"], k=req.top_k, use_graph=req.use_graph, tag="api")
    return AskResponse(answer=r["answer"], abstained=r["abstained"], citations=r["citations"],
                       invalid_citations=r["invalid_citations"],
                       sources=[Source(**{k: s[k] for k in ("id", "title", "section", "sensitivity", "score")}) for s in r["sources"]],
                       graph_ids=r["graph_ids"], role=user["role"], prompt_tokens=r["prompt_tokens"],
                       completion_tokens=r["completion_tokens"], cost_usd=r["cost_usd"], latency_ms=r["latency_ms"]["total"])


@app.get("/sources/{source_id}", response_model=SourceResponse, responses=ERRORS, summary="Текст джерела за id")
def get_source(source_id: str = PathParam(..., min_length=3, max_length=120, pattern=r"^[\w:.\-]+$"),
               user: dict = Depends(auth)):
    c = STATE["rag"].source_index.get(source_id)
    if c is None:
        raise HTTPException(404, f"Джерело {source_id} не знайдено")
    if LEVEL[c["sensitivity"]] > LEVEL[ROLE_MAX[user["role"]]]:
        raise HTTPException(403, f"Роль {user['role']} не має доступу до джерел рівня {c['sensitivity']}")
    return SourceResponse(id=source_id, title=c["title"], section=c["section"], sensitivity=c["sensitivity"],
                          source_url=c["source_url"], date=c.get("date"), text=c["text"])
