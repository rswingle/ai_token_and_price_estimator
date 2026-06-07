"""CSV output for the cost estimator.

Single file with a 'section' column. Sections: token_estimate,
recommendation, pricing, providers, excluded.
"""
from __future__ import annotations

import csv
import io
from typing import Any, Sequence

from .cost_engine import CostLine
from .recommender import ScoredLine
from .token_estimator import TokenEstimate


COLUMNS: tuple[str, ...] = (
    "section",
    "rank",
    "kind",
    "provider",
    "model",
    "input_usd_per_1m",
    "output_usd_per_1m",
    "total_cost_usd",
    "weighted_score",
    "cost_score",
    "capability_score",
    "context_score",
    "latency_score",
    "fits_context",
    "context_window",
    "model_count",
    "task_kind",
    "multiplier",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt",
    "reason",
    "note",
)


def _money(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"${float(value):.6f}"


def _score(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _int(value: int | None) -> str:
    if value is None:
        return ""
    return f"{int(value):,}"


def _float(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _token_estimate_rows(prompt: str, est: TokenEstimate) -> list[dict[str, str]]:
    return [
        {
            "section": "token_estimate",
            "task_kind": est.task_kind,
            "multiplier": _float(est.multiplier_used),
            "input_tokens": _int(est.input_tokens),
            "output_tokens": _int(est.output_tokens_estimated),
            "total_tokens": _int(est.total_tokens),
            "prompt": prompt,
        }
    ]


def _recommendation_rows(recs: Sequence[ScoredLine]) -> list[dict[str, str]]:
    if not recs:
        return []
    out: list[dict[str, str]] = []
    for i, r in enumerate(recs, 1):
        cl = r.cost_line
        out.append(
            {
                "section": "recommendation",
                "rank": str(i),
                "kind": "best" if i == 1 else "alternative",
                "provider": cl.provider,
                "model": cl.model,
                "input_usd_per_1m": _money(cl.input_cost_per_1m),
                "output_usd_per_1m": _money(cl.output_cost_per_1m),
                "total_cost_usd": _money(cl.total_cost),
                "weighted_score": _score(r.weighted_score),
                "cost_score": _score(r.cost_score),
                "capability_score": _score(r.capability_score),
                "context_score": _score(r.context_score),
                "latency_score": _score(r.latency_score),
                "fits_context": str(cl.fits_context_window),
                "context_window": _int(cl.context_window),
                "reason": r.reason,
            }
        )
    return out


def _pricing_rows(ranked: Sequence[CostLine]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, c in enumerate(ranked, 1):
        out.append(
            {
                "section": "pricing",
                "rank": str(i),
                "provider": c.provider,
                "model": c.model,
                "input_usd_per_1m": _money(c.input_cost_per_1m),
                "output_usd_per_1m": _money(c.output_cost_per_1m),
                "total_cost_usd": _money(c.total_cost),
                "context_window": _int(c.context_window),
            }
        )
    return out


def _providers_rows(ranked: Sequence[CostLine]) -> list[dict[str, str]]:
    by_provider: dict[str, list[CostLine]] = {}
    for c in ranked:
        by_provider.setdefault(c.provider, []).append(c)
    out: list[dict[str, str]] = []
    for i, (provider, lines) in enumerate(sorted(by_provider.items()), 1):
        total = sum(c.total_cost for c in lines)
        out.append(
            {
                "section": "providers",
                "rank": str(i),
                "provider": provider,
                "model_count": str(len(lines)),
                "total_cost_usd": _money(total),
            }
        )
    return out


def _excluded_rows(excluded: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, r in enumerate(excluded, 1):
        reasons = r.get("reasons") or []
        if isinstance(reasons, list):
            reason = "; ".join(str(x) for x in reasons)
        else:
            reason = str(reasons)
        out.append(
            {
                "section": "excluded",
                "rank": str(i),
                "provider": str(r.get("provider", "")),
                "model": str(r.get("model", "")),
                "reason": reason,
            }
        )
    return out


def _collect_rows(
    prompt: str,
    est: TokenEstimate,
    ranked: Sequence[CostLine],
    recs: Sequence[ScoredLine],
    excluded: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(_token_estimate_rows(prompt, est))
    rows.extend(_recommendation_rows(recs))
    rows.extend(_pricing_rows(ranked))
    rows.extend(_providers_rows(ranked))
    if excluded:
        rows.extend(_excluded_rows(excluded))
    else:
        rows.append({"section": "excluded", "note": "No records were excluded."})
    return rows


def build_csv(
    prompt: str,
    est: TokenEstimate,
    pricing_meta: dict[str, Any],
    kept: Sequence[dict[str, Any]],
    excluded: Sequence[dict[str, Any]],
    ranked: Sequence[CostLine],
    recs: Sequence[ScoredLine],
) -> str:
    rows = _collect_rows(prompt, est, ranked, recs, excluded)
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=list(COLUMNS), extrasaction="ignore"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()
