"""Клієнт LLM (OpenAI-сумісний API; за замовчуванням Groq) з кешем, повторами при 429 і обліком токенів/вартості.

Змінні середовища (файл .env у корені проєкту, НЕ комітиться):
  GROQ_API_KEY   — ключ Groq (https://console.groq.com/keys)
  LLM_BASE_URL   — за замовчуванням https://api.groq.com/openai/v1
  GEN_MODEL      — генератор, за замовчуванням openai/gpt-oss-120b
  JUDGE_MODEL    — LLM-суддя (інша модель), за замовчуванням openai/gpt-oss-20b
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, SEED  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", encoding="utf-8-sig")      # utf-8-sig: файл, збережений Блокнотом із BOM
except Exception:  # noqa: BLE001
    pass

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
GEN_MODEL = os.getenv("GEN_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")

# $ за 1 млн токенів (вхід, вихід) — ціни Groq з https://console.groq.com/docs/models
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}
CACHE_DIR = DATA / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DailyLimitError(RuntimeError):
    """Вичерпано денний ліміт безкоштовного плану — запустіть пізніше (готові відповіді збережено в кеші)."""


class LLMError(RuntimeError):
    pass


def cost_usd(model, prompt_tokens, completion_tokens):
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            env = ROOT / ".env"
            hint = "файлу немає" if not env.exists() else "у файлі немає рядка GROQ_API_KEY=..."
            raise LLMError(f"Немає GROQ_API_KEY ({env}: {hint}). Див. .env.example")
        from openai import OpenAI
        _client = OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=120)
    return _client


def chat(messages, model=None, max_tokens=700, temperature=0.0, json_mode=False, use_cache=True):
    """Повертає dict: text, prompt_tokens, completion_tokens, latency_ms, cost_usd, model, cached."""
    model = model or GEN_MODEL
    key_src = json.dumps({"m": model, "msg": messages, "t": temperature, "mt": max_tokens, "j": json_mode},
                         ensure_ascii=False, sort_keys=True)
    path = CACHE_DIR / (hashlib.sha1(key_src.encode("utf-8")).hexdigest() + ".json")
    if use_cache and path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["cached"] = True
        return rec

    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, seed=SEED)
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"          # менше «прихованих» токенів міркувань
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    client = _get_client()
    for attempt in range(8):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            status = getattr(e, "status_code", None)
            if status == 429 or "rate limit" in msg.lower():
                if re.search(r"per day|TPD|RPD", msg):
                    raise DailyLimitError(msg[:300]) from e
                m = re.search(r"try again in ([\d.]+)(ms|s|m)", msg)
                wait = 10.0
                if m:
                    v, unit = float(m.group(1)), m.group(2)
                    wait = v / 1000 if unit == "ms" else v * 60 if unit == "m" else v
                wait = min(max(wait + 1, 2), 90)
                print(f"  … ліміт Groq (хвилинний), пауза {wait:.0f} с")
                time.sleep(wait)
                continue
            if status == 400 and "reasoning_effort" in msg:
                kwargs.pop("reasoning_effort", None)
                continue
            if status == 400 and "response_format" in msg:
                kwargs.pop("response_format", None)
                continue
            if status and status >= 500 and attempt < 3:
                time.sleep(5)
                continue
            raise LLMError(msg[:500]) from e
    else:
        raise LLMError("Не вдалося отримати відповідь LLM після повторів")
    latency = (time.perf_counter() - t0) * 1000
    usage = resp.usage
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    rec = {"text": (resp.choices[0].message.content or "").strip(), "prompt_tokens": pt, "completion_tokens": ct,
           "latency_ms": latency, "cost_usd": cost_usd(model, pt, ct), "model": model, "cached": False}
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def parse_json(text):
    """Дістає перший JSON-об'єкт із відповіді моделі."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
