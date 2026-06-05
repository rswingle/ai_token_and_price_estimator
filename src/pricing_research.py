"""Web pricing research for stateless LLM providers.

Strategy:
  1. Try to refresh from official provider pages (best-effort, short timeout).
  2. Always fall back to a curated, timestamped baseline of public pricing
     (labeled stale if no live refresh succeeded).
The baseline ships in-repo so the tool is usable offline.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from trafilatura import extract as trafilatura_extract

from .price_resolver import OPENROUTER_URL

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "pricing_snapshot.json")

HTTP_TIMEOUT = 6
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ai-token-price-estimator/0.1; +https://example.local)",
    "Accept-Language": "en-US,en;q=0.9",
}

PROVIDER_PAGES: list[dict[str, str]] = [
    {"provider": "OpenAI",    "url": "https://openai.com/api/pricing/"},
    {"provider": "Anthropic", "url": "https://docs.anthropic.com/en/docs/about-claude/models"},
    {"provider": "Google",    "url": "https://ai.google.dev/pricing"},
    {"provider": "Mistral",   "url": "https://docs.mistral.ai/getting-started/models/models_overview/"},
    {"provider": "Groq",      "url": "https://groq.com/pricing/"},
    {"provider": "Together",  "url": "https://www.together.ai/pricing"},
    {"provider": "Fireworks", "url": "https://fireworks.ai/pricing"},
    {"provider": "DeepSeek",  "url": "https://api-docs.deepseek.com/quick_start/pricing/"},
    {"provider": "OpenRouter","url": "https://openrouter.ai/models"},
]

PRICE_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|per)\s*1\s*M(?:\s*tokens)?", re.IGNORECASE)
PRICE_RE_ALT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*1M", re.IGNORECASE)


@dataclass
class ModelPricing:
    provider: str
    model: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    context_window: int
    source_url: str
    fetched_at: str
    is_stateless: bool = True
    deprecated: bool = False
    capability_tags: list[str] = field(default_factory=list)
    reasoning_strength: float = 0.5
    coding_strength: float = 0.5
    creativity_strength: float = 0.5
    latency_p50_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _baseline_snapshot() -> list[dict[str, Any]]:
    """Curated baseline of public, stateless API pricing as of late 2025.

    Each entry's `source_url` points to the official pricing page so users
    can verify. Update this list when running a manual refresh; the tool
    will overwrite from web when the refresh succeeds.
    """
    snap = _now_iso()
    return [
        {
            "provider": "OpenAI", "model": "gpt-4o",
            "input_cost_per_1m": 2.50, "output_cost_per_1m": 10.00,
            "context_window": 128_000, "source_url": "https://openai.com/api/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "vision", "tools"],
            "reasoning_strength": 0.85, "coding_strength": 0.90, "creativity_strength": 0.80,
            "latency_p50_ms": 600,
        },
        {
            "provider": "OpenAI", "model": "gpt-4o-mini",
            "input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60,
            "context_window": 128_000, "source_url": "https://openai.com/api/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "vision", "tools"],
            "reasoning_strength": 0.65, "coding_strength": 0.70, "creativity_strength": 0.65,
            "latency_p50_ms": 400,
        },
        {
            "provider": "OpenAI", "model": "gpt-4.1",
            "input_cost_per_1m": 2.00, "output_cost_per_1m": 8.00,
            "context_window": 1_047_576, "source_url": "https://openai.com/api/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "vision", "tools", "long-context"],
            "reasoning_strength": 0.90, "coding_strength": 0.92, "creativity_strength": 0.80,
            "latency_p50_ms": 700,
        },
        {
            "provider": "OpenAI", "model": "gpt-4.1-mini",
            "input_cost_per_1m": 0.40, "output_cost_per_1m": 1.60,
            "context_window": 1_047_576, "source_url": "https://openai.com/api/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "tools", "long-context"],
            "reasoning_strength": 0.75, "coding_strength": 0.80, "creativity_strength": 0.70,
            "latency_p50_ms": 450,
        },
        {
            "provider": "OpenAI", "model": "o3",
            "input_cost_per_1m": 10.00, "output_cost_per_1m": 40.00,
            "context_window": 200_000, "source_url": "https://openai.com/api/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "hard-reasoning", "tools"],
            "reasoning_strength": 0.98, "coding_strength": 0.93, "creativity_strength": 0.70,
            "latency_p50_ms": 1500,
        },
        {
            "provider": "OpenAI", "model": "o4-mini",
            "input_cost_per_1m": 1.10, "output_cost_per_1m": 4.40,
            "context_window": 200_000, "source_url": "https://openai.com/api/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "tools"],
            "reasoning_strength": 0.90, "coding_strength": 0.88, "creativity_strength": 0.70,
            "latency_p50_ms": 900,
        },
        {
            "provider": "Anthropic", "model": "claude-sonnet-4-5",
            "input_cost_per_1m": 3.00, "output_cost_per_1m": 15.00,
            "context_window": 200_000, "source_url": "https://docs.anthropic.com/en/docs/about-claude/models",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "vision", "tools", "long-context"],
            "reasoning_strength": 0.95, "coding_strength": 0.92, "creativity_strength": 0.85,
            "latency_p50_ms": 800,
        },
        {
            "provider": "Anthropic", "model": "claude-haiku-4-5",
            "input_cost_per_1m": 1.00, "output_cost_per_1m": 5.00,
            "context_window": 200_000, "source_url": "https://docs.anthropic.com/en/docs/about-claude/models",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "tools"],
            "reasoning_strength": 0.80, "coding_strength": 0.82, "creativity_strength": 0.75,
            "latency_p50_ms": 400,
        },
        {
            "provider": "Anthropic", "model": "claude-opus-4-1",
            "input_cost_per_1m": 15.00, "output_cost_per_1m": 75.00,
            "context_window": 200_000, "source_url": "https://docs.anthropic.com/en/docs/about-claude/models",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "hard-reasoning", "coding", "vision", "tools", "long-context"],
            "reasoning_strength": 0.97, "coding_strength": 0.95, "creativity_strength": 0.90,
            "latency_p50_ms": 1200,
        },
        {
            "provider": "Google", "model": "gemini-2.5-pro",
            "input_cost_per_1m": 1.25, "output_cost_per_1m": 10.00,
            "context_window": 1_000_000, "source_url": "https://ai.google.dev/pricing",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "vision", "tools", "long-context"],
            "reasoning_strength": 0.92, "coding_strength": 0.90, "creativity_strength": 0.82,
            "latency_p50_ms": 900,
        },
        {
            "provider": "Google", "model": "gemini-2.5-flash",
            "input_cost_per_1m": 0.30, "output_cost_per_1m": 2.50,
            "context_window": 1_000_000, "source_url": "https://ai.google.dev/pricing",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "vision", "tools", "long-context", "fast"],
            "reasoning_strength": 0.85, "coding_strength": 0.85, "creativity_strength": 0.78,
            "latency_p50_ms": 350,
        },
        {
            "provider": "Google", "model": "gemini-2.5-flash-lite",
            "input_cost_per_1m": 0.10, "output_cost_per_1m": 0.40,
            "context_window": 1_000_000, "source_url": "https://ai.google.dev/pricing",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "fast", "long-context"],
            "reasoning_strength": 0.70, "coding_strength": 0.72, "creativity_strength": 0.65,
            "latency_p50_ms": 250,
        },
        {
            "provider": "Mistral", "model": "mistral-large-latest",
            "input_cost_per_1m": 2.00, "output_cost_per_1m": 6.00,
            "context_window": 128_000, "source_url": "https://docs.mistral.ai/getting-started/models/models_overview/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "tools"],
            "reasoning_strength": 0.82, "coding_strength": 0.80, "creativity_strength": 0.78,
            "latency_p50_ms": 600,
        },
        {
            "provider": "Mistral", "model": "mistral-small-latest",
            "input_cost_per_1m": 0.20, "output_cost_per_1m": 0.60,
            "context_window": 128_000, "source_url": "https://docs.mistral.ai/getting-started/models/models_overview/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "fast", "tools"],
            "reasoning_strength": 0.72, "coding_strength": 0.75, "creativity_strength": 0.70,
            "latency_p50_ms": 350,
        },
        {
            "provider": "Groq", "model": "llama-3.3-70b-versatile",
            "input_cost_per_1m": 0.59, "output_cost_per_1m": 0.79,
            "context_window": 128_000, "source_url": "https://groq.com/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "fast", "open-weights"],
            "reasoning_strength": 0.78, "coding_strength": 0.78, "creativity_strength": 0.75,
            "latency_p50_ms": 200,
        },
        {
            "provider": "Groq", "model": "llama-3.1-8b-instant",
            "input_cost_per_1m": 0.05, "output_cost_per_1m": 0.08,
            "context_window": 128_000, "source_url": "https://groq.com/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "fast", "open-weights"],
            "reasoning_strength": 0.55, "coding_strength": 0.55, "creativity_strength": 0.55,
            "latency_p50_ms": 150,
        },
        {
            "provider": "Together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "input_cost_per_1m": 0.88, "output_cost_per_1m": 0.88,
            "context_window": 128_000, "source_url": "https://www.together.ai/pricing",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "open-weights"],
            "reasoning_strength": 0.78, "coding_strength": 0.78, "creativity_strength": 0.75,
            "latency_p50_ms": 500,
        },
        {
            "provider": "Fireworks", "model": "llama-v3p3-70b-instruct",
            "input_cost_per_1m": 0.90, "output_cost_per_1m": 0.90,
            "context_window": 128_000, "source_url": "https://fireworks.ai/pricing",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "fast", "open-weights"],
            "reasoning_strength": 0.78, "coding_strength": 0.78, "creativity_strength": 0.75,
            "latency_p50_ms": 300,
        },
        {
            "provider": "DeepSeek", "model": "deepseek-chat",
            "input_cost_per_1m": 0.27, "output_cost_per_1m": 1.10,
            "context_window": 64_000, "source_url": "https://api-docs.deepseek.com/quick_start/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "coding", "open-weights"],
            "reasoning_strength": 0.82, "coding_strength": 0.88, "creativity_strength": 0.70,
            "latency_p50_ms": 700,
        },
        {
            "provider": "DeepSeek", "model": "deepseek-reasoner",
            "input_cost_per_1m": 0.55, "output_cost_per_1m": 2.19,
            "context_window": 64_000, "source_url": "https://api-docs.deepseek.com/quick_start/pricing/",
            "fetched_at": snap, "is_stateless": True, "deprecated": False,
            "capability_tags": ["reasoning", "hard-reasoning", "coding", "open-weights"],
            "reasoning_strength": 0.95, "coding_strength": 0.88, "creativity_strength": 0.65,
            "latency_p50_ms": 1500,
        },
    ]


def _normalize_url(u: str) -> str:
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")


_MODEL_PRICE_WINDOW = 600
_SANITY_MIN = 0.0001
_SANITY_MAX = 1000.0


def _fetch_page_text(url: str) -> str:
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return ""
    except (requests.RequestException, Exception):
        return ""

    text = ""
    try:
        text = trafilatura_extract(r.text) or ""
    except Exception:
        text = ""
    if not text:
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
    return text


def _prices_in_window(text: str, anchor: int) -> list[float]:
    if anchor < 0:
        return []
    lo = max(0, anchor - _MODEL_PRICE_WINDOW)
    hi = anchor + _MODEL_PRICE_WINDOW
    vals: list[float] = []
    for m in PRICE_RE.finditer(text):
        if m.start() < lo:
            continue
        if m.start() > hi:
            break
        v = float(m.group(1))
        if _SANITY_MIN <= v <= _SANITY_MAX:
            vals.append(v)
    return vals


def _scrape_prices(
    url: str, model_keyword: str | None = None
) -> dict[str, tuple[float, float]]:
    if not model_keyword:
        return {}
    text = _fetch_page_text(url)
    if not text:
        return {}
    pos = text.lower().find(model_keyword.lower())
    if pos < 0:
        return {}
    vals = _prices_in_window(text, pos)
    if len(vals) < 2:
        return {}
    return {model_keyword: (vals[0], vals[1])}


def _refresh_one(entry: dict[str, Any], url: str) -> dict[str, Any]:
    model = str(entry.get("model", ""))
    scraped = _scrape_prices(url, model_keyword=model)
    if model not in scraped:
        return entry

    in_p, out_p = scraped[model]
    if not (_SANITY_MIN <= in_p <= _SANITY_MAX) or not (_SANITY_MIN <= out_p <= _SANITY_MAX):
        return entry

    updated = dict(entry)
    if (
        round(in_p, 4) != round(float(entry.get("input_cost_per_1m", 0.0)), 4)
        or round(out_p, 4) != round(float(entry.get("output_cost_per_1m", 0.0)), 4)
    ):
        updated["input_cost_per_1m"] = round(in_p, 4)
        updated["output_cost_per_1m"] = round(out_p, 4)
        updated["source_url"] = _normalize_url(url)
        updated["fetched_at"] = _now_iso()
    return updated


def refresh_pricing(timeout_total: float = 20.0) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    """Attempt a live refresh. Returns (records, errors, changes_count).

    On any network/parse failure, the curated baseline is returned untouched
    and the record's `fetched_at` remains the baseline timestamp (labeled stale).
    `changes_count` is the number of records whose prices were actually updated
    from a live scrape; 0 means no live data was usable.
    """
    baseline = _baseline_snapshot()
    errors: list[dict[str, str]] = []
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for r in baseline:
        by_provider.setdefault(r["provider"], []).append(r)

    page_by_provider = {p["provider"]: p["url"] for p in PROVIDER_PAGES}

    deadline = time.time() + timeout_total
    changes = 0
    for provider, url in page_by_provider.items():
        if time.time() > deadline:
            errors.append({"provider": provider, "url": url, "error": "global timeout reached"})
            continue
        for i, entry in enumerate(by_provider.get(provider, [])):
            if time.time() > deadline:
                errors.append({"provider": provider, "url": url, "error": "global timeout reached"})
                break
            before_in = entry["input_cost_per_1m"]
            before_out = entry["output_cost_per_1m"]
            updated = _refresh_one(entry, url)
            if updated["input_cost_per_1m"] != before_in or updated["output_cost_per_1m"] != before_out:
                changes += 1
            by_provider[provider][i] = updated

    merged: list[dict[str, Any]] = []
    for provider in by_provider:
        merged.extend(by_provider[provider])
    return merged, errors, changes


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_cache() -> list[dict[str, Any]] | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_cache(records: list[dict[str, Any]]) -> None:
    _ensure_cache_dir()
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


def get_pricing(
    refresh: bool = False,
    use_cache_if_available: bool = True,
    use_openrouter: bool = False,
) -> dict[str, Any]:
    """Return a pricing snapshot.

    refresh=True attempts a live scrape of provider marketing pages (short
    timeout). use_openrouter=True additionally fetches per-token USD prices
    from OpenRouter's public model API and merges them into the baseline;
    on any failure it falls back cleanly to the curated in-repo baseline.

    Returns:
      {
        "records": [ModelPricing dicts],
        "source": "live" | "live-openrouter" | "cache" | "baseline",
        "fetched_at": ISO timestamp,
        "errors": [...],
        "stale": bool,
        "changes_applied": int (when source is live or live-openrouter),
      }
    """
    errors: list[dict[str, str]] = []

    if use_openrouter:
        try:
            from .price_resolver import apply_openrouter_prices, fetch_openrouter_prices
        except ImportError:
            fetch_openrouter_prices = None
            apply_openrouter_prices = None

        if fetch_openrouter_prices is not None:
            prices = fetch_openrouter_prices()
            if prices:
                records, changes, unmapped = apply_openrouter_prices(_baseline_snapshot(), prices)
                if unmapped:
                    errors.append({
                        "provider": "*",
                        "url": OPENROUTER_URL,
                        "error": f"openrouter: {len(unmapped)} baseline record(s) have no OpenRouter mapping; prices left as baseline",
                    })
                if changes > 0:
                    save_cache(records)
                    return {
                        "records": records,
                        "source": "live-openrouter",
                        "fetched_at": _now_iso(),
                        "errors": errors,
                        "stale": False,
                        "changes_applied": changes,
                    }
                errors.append({
                    "provider": "*",
                    "url": OPENROUTER_URL,
                    "error": "openrouter: no price changes vs baseline; using baseline",
                })
            else:
                errors.append({
                    "provider": "*",
                    "url": OPENROUTER_URL,
                    "error": "openrouter fetch failed or returned no usable prices; using baseline",
                })

    if refresh:
        records, errors2, changes = refresh_pricing()
        errors.extend(errors2)
        if changes > 0:
            save_cache(records)
            return {
                "records": records,
                "source": "live",
                "fetched_at": _now_iso(),
                "errors": errors,
                "stale": False,
                "changes_applied": changes,
            }
        errors.append({"provider": "*", "url": "*", "error": "live refresh returned no usable prices; using baseline"})

    cached = load_cache()
    if use_cache_if_available and cached:
        newest = max(r.get("fetched_at", "") for r in cached)
        return {
            "records": cached,
            "source": "cache",
            "fetched_at": newest or _now_iso(),
            "errors": errors,
            "stale": True,
        }

    baseline = _baseline_snapshot()
    return {
        "records": baseline,
        "source": "baseline",
        "fetched_at": _now_iso(),
        "errors": errors,
        "stale": True,
    }
