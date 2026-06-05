"""Stateless-only filter.

Excludes chat/session products and any service with stateful or
memory-based billing. Keeps only pay-per-token inference APIs.
"""
from __future__ import annotations

import re
from typing import Any

SESSION_PATTERNS = re.compile(
    r"(?<![a-z0-9])("
    r"chatgpt[\s-]+(?:plus|team|pro)|"
    r"claude[\s-]+(?:pro|team|max|free|ai|chat)|"
    r"perplexity[\s-]+(?:pro|chat|sonnet)|"
    r"gemini[\s-]+advanced|"
    r"(?:microsoft[\s-]+)?copilot[\s-]+pro|"
    r"poe[\s-]+pro|character\.ai|character[\s-]+ai|replika|pi\.ai|"
    r"chat[\s-]*(?:session|product|app)|"
    r"session[\s-]*based|stateful|memory[\s-]*based[\s-]+billing"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)

DEPRECATED_PATTERNS = re.compile(
    r"\b(gpt-3\.5-turbo-0613|gpt-3\.5-turbo-0301|text-davinci-|"
    r"claude-2(?:\.1)?|claude-instant|claude-3-haiku-20240307|"
    r"gemini-pro-1\.0|gemini-1\.0-pro|"
    r"mistral-7b-instruct|mixtral-8x7b-instruct(?:$|\s))\b",
    re.IGNORECASE,
)


def is_session_based(record: dict[str, Any]) -> bool:
    """True if a record describes a chat/UI/session product (not a stateless API)."""
    if record.get("is_stateless") is False:
        return True
    haystack = " ".join(
        str(record.get(k, "")) for k in ("provider", "model", "source_url", "capability_tags")
    )
    return bool(SESSION_PATTERNS.search(haystack))


def is_deprecated(record: dict[str, Any]) -> bool:
    if record.get("deprecated") is True:
        return True
    return bool(DEPRECATED_PATTERNS.search(str(record.get("model", ""))))


def has_required_pricing(record: dict[str, Any]) -> bool:
    try:
        inp = float(record.get("input_cost_per_1m", -1))
        out = float(record.get("output_cost_per_1m", -1))
    except (TypeError, ValueError):
        return False
    if inp < 0 or out < 0:
        return False
    if inp > 1000 or out > 1000:
        return False
    return True


def filter_records(
    records: list[dict[str, Any]],
    *,
    include_deprecated: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (kept, excluded_with_reason)."""
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for r in records:
        reason: list[str] = []
        if is_session_based(r):
            reason.append("session-based or chat product")
        if not include_deprecated and is_deprecated(r):
            reason.append("deprecated model")
        if not has_required_pricing(r):
            reason.append("missing or invalid pricing")
        if reason:
            excluded.append({
                "provider": r.get("provider"),
                "model": r.get("model"),
                "reasons": reason,
            })
        else:
            kept.append(r)
    return kept, excluded
