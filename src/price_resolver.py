"""Opt-in price resolver using OpenRouter's public model API.

Merges only prices back into the internal baseline; capability metadata is
preserved. Per-vendor SDKs (OpenAI, Anthropic) do not expose prices, so
OpenRouter is the only public programmatic cross-provider source.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
HTTP_TIMEOUT = 10
HTTP_HEADERS = {
    "User-Agent": "ai-token-price-estimator/0.1 (+opt-in-openrouter)",
    "Accept": "application/json",
}

_SANITY_MIN = 0.0001
_SANITY_MAX = 1000.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_OPENROUTER_MAP: dict[tuple[str, str], str] = {
    ("OpenAI", "gpt-4o"):              "openai/gpt-4o",
    ("OpenAI", "gpt-4o-mini"):         "openai/gpt-4o-mini",
    ("OpenAI", "gpt-4.1"):             "openai/gpt-4.1",
    ("OpenAI", "gpt-4.1-mini"):        "openai/gpt-4.1-mini",
    ("OpenAI", "o3"):                  "openai/o3",
    ("OpenAI", "o4-mini"):             "openai/o4-mini",
    ("Anthropic", "claude-sonnet-4-5"): "anthropic/claude-sonnet-4.5",
    ("Anthropic", "claude-haiku-4-5"):   "anthropic/claude-haiku-4.5",
    ("Anthropic", "claude-opus-4-1"):    "anthropic/claude-opus-4-1",
    ("Google", "gemini-2.5-pro"):        "google/gemini-2.5-pro",
    ("Google", "gemini-2.5-flash"):      "google/gemini-2.5-flash",
    ("Google", "gemini-2.5-flash-lite"): "google/gemini-2.5-flash-lite",
    ("Mistral", "mistral-large-latest"): "mistralai/mistral-large-latest",
    ("Mistral", "mistral-small-latest"): "mistralai/mistral-small-latest",
    ("DeepSeek", "deepseek-chat"):       "deepseek/deepseek-chat",
    ("DeepSeek", "deepseek-reasoner"):   "deepseek/deepseek-reasoner",
    ("Groq", "llama-3.3-70b-versatile"): "meta-llama/llama-3.3-70b-versatile",
    ("Groq", "llama-3.1-8b-instant"):    "meta-llama/llama-3.1-8b-instant",
}


def _parse_price(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    per_million = v * 1_000_000
    if not (_SANITY_MIN <= per_million <= _SANITY_MAX):
        return None
    return per_million


def fetch_openrouter_prices(timeout: float = HTTP_TIMEOUT) -> dict[str, tuple[float, float]] | None:
    try:
        r = requests.get(OPENROUTER_URL, headers=HTTP_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
    except (requests.RequestException, ValueError, Exception):
        return None

    out: dict[str, tuple[float, float]] = {}
    for m in data.get("data", []):
        mid = m.get("id")
        pricing = m.get("pricing") or {}
        if not isinstance(mid, str) or "/" not in mid:
            continue
        p_in = _parse_price(pricing.get("prompt"))
        p_out = _parse_price(pricing.get("completion"))
        if p_in is None or p_out is None:
            continue
        out[mid] = (round(p_in, 4), round(p_out, 4))
    return out if out else None


def apply_openrouter_prices(
    baseline: list[dict[str, Any]],
    prices: dict[str, tuple[float, float]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    updated: list[dict[str, Any]] = []
    changes = 0
    unmapped: list[dict[str, Any]] = []
    for rec in baseline:
        key = (rec.get("provider", ""), rec.get("model", ""))
        or_id = _OPENROUTER_MAP.get(key)
        if not or_id or or_id not in prices:
            unmapped.append({"provider": key[0], "model": key[1], "openrouter_id": or_id})
            updated.append(rec)
            continue
        new_in, new_out = prices[or_id]
        old_in = round(float(rec.get("input_cost_per_1m", 0.0)), 4)
        old_out = round(float(rec.get("output_cost_per_1m", 0.0)), 4)
        if new_in != old_in or new_out != old_out:
            new_rec = dict(rec)
            new_rec["input_cost_per_1m"] = new_in
            new_rec["output_cost_per_1m"] = new_out
            new_rec["source_url"] = "https://openrouter.ai/models"
            new_rec["fetched_at"] = _now_iso()
            updated.append(new_rec)
            changes += 1
        else:
            updated.append(rec)
    return updated, changes, unmapped

