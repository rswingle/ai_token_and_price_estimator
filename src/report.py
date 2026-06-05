"""Output formatters: human-readable report and machine-readable JSON."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .cost_engine import CostLine
from .recommender import ScoredLine
from .token_estimator import TokenEstimate


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_json(
    prompt: str,
    estimate: TokenEstimate,
    pricing_meta: dict[str, Any],
    kept_records: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    ranked_lines: list[CostLine],
    recommendations: list[ScoredLine],
) -> dict[str, Any]:
    best = recommendations[0] if recommendations else None
    return {
        "generated_at": _now_iso(),
        "project_prompt": prompt,
        "token_estimate": estimate.to_dict(),
        "pricing_source": {
            "source": pricing_meta.get("source"),
            "stale": pricing_meta.get("stale"),
            "fetched_at": pricing_meta.get("fetched_at"),
            "errors": pricing_meta.get("errors", []),
        },
        "providers": kept_records,
        "excluded": excluded,
        "cost_comparison": [ln.to_dict() for ln in ranked_lines],
        "recommendation": (
            {
                "best_model": best.cost_line.model,
                "best_provider": best.cost_line.provider,
                "best_total_cost_usd": best.cost_line.total_cost,
                "best_weighted_score": best.weighted_score,
                "reason": best.reason,
                "alternatives": [
                    {
                        "rank": s.rank,
                        "provider": s.cost_line.provider,
                        "model": s.cost_line.model,
                        "total_cost_usd": s.cost_line.total_cost,
                        "weighted_score": s.weighted_score,
                        "reason": s.reason,
                    }
                    for s in recommendations[1:]
                ],
            }
            if best
            else None
        ),
    }


def _hr(char: str = "─", width: int = 78) -> str:
    return char * width


def _fmt_money(x: float) -> str:
    if x == 0:
        return "$0.000000"
    if abs(x) < 0.0001:
        return f"${x:.8f}"
    return f"${x:.6f}"


def build_human(
    prompt: str,
    estimate: TokenEstimate,
    pricing_meta: dict[str, Any],
    excluded: list[dict[str, Any]],
    ranked_lines: list[CostLine],
    recommendations: list[ScoredLine],
) -> str:
    out: list[str] = []
    out.append(_hr("═"))
    out.append("  AI TOKEN & PRICE ESTIMATOR")
    out.append(_hr("═"))
    out.append(f"  Generated:        {_now_iso()}")
    out.append(f"  Pricing source:   {pricing_meta.get('source')}  (stale={pricing_meta.get('stale')})")
    out.append(f"  Pricing fetched:  {pricing_meta.get('fetched_at')}")
    if pricing_meta.get("errors"):
        out.append(f"  Pricing errors:   {len(pricing_meta['errors'])} (see JSON output for details)")
    out.append("")

    out.append("  PROJECT PROMPT")
    out.append("  " + _hr("─", 76))
    for line in prompt.splitlines() or [prompt]:
        out.append(f"  > {line}")
    out.append("")

    out.append("  TOKEN ESTIMATE")
    out.append("  " + _hr("─", 76))
    out.append(f"  Task kind:                {estimate.task_kind}")
    out.append(f"  Input tokens:             {estimate.input_tokens}")
    out.append(f"  Output tokens (est.):     {estimate.output_tokens_estimated}")
    out.append(f"  Reasoning buffer:         {estimate.reasoning_buffer_tokens}")
    out.append(f"  Total tokens:             {estimate.total_tokens}")
    out.append(f"  Multiplier used:          {estimate.multiplier_used}x")
    out.append("")
    out.append("  Notes:")
    for n in estimate.notes:
        out.append(f"    - {n}")
    out.append("")

    out.append("  PRICING COMPARISON (ranked by total cost)")
    out.append("  " + _hr("─", 76))
    out.append(f"  {'Rank':<4} {'Provider':<11} {'Model':<33} {'Cost USD':>12}  {'Ctx fits':<8}")
    for i, ln in enumerate(ranked_lines, 1):
        out.append(
            f"  {i:<4} {ln.provider:<11} {ln.model:<33} {_fmt_money(ln.total_cost):>12}  "
            f"{('yes' if ln.fits_context_window else 'NO'):<8}"
        )
    out.append("")

    out.append("  RECOMMENDATION (weighted scoring)")
    out.append("  " + _hr("─", 76))
    out.append("  Weights: cost=0.40 capability=0.30 context=0.20 latency=0.10")
    out.append("")
    if recommendations:
        best = recommendations[0]
        out.append(f"  ★ BEST: {best.cost_line.provider} / {best.cost_line.model}")
        out.append(f"     Total cost:   {_fmt_money(best.cost_line.total_cost)}")
        out.append(f"     Score:        {best.weighted_score:.3f}")
        out.append(f"     Reason:       {best.reason}")
        out.append("")
        if len(recommendations) > 1:
            out.append("  Alternatives:")
            for s in recommendations[1:]:
                out.append(
                    f"     #{s.rank} {s.cost_line.provider}/{s.cost_line.model}  "
                    f"cost={_fmt_money(s.cost_line.total_cost)}  score={s.weighted_score:.3f}"
                )
        out.append("")
    else:
        out.append("  No recommendations available (no providers matched).")
        out.append("")

    if excluded:
        out.append("  EXCLUDED (filtered out)")
        out.append("  " + _hr("─", 76))
        for e in excluded[:20]:
            out.append(f"    - {e.get('provider')}/{e.get('model')}: {', '.join(e.get('reasons', []))}")
        if len(excluded) > 20:
            out.append(f"    ... and {len(excluded) - 20} more")
        out.append("")

    out.append("  WARNINGS & ASSUMPTIONS")
    out.append("  " + _hr("─", 76))
    if pricing_meta.get("stale"):
        out.append("  * Pricing data may be stale. Run with --refresh to attempt a live update.")
    if not estimate.notes or "user-provided" not in " ".join(estimate.notes).lower():
        out.append("  * Output token count is HEURISTIC (multiplier-based). Override with --target-output for accuracy.")
    out.append("  * Per-spec, only stateless API providers are considered. Chat/UI/session products are excluded.")
    out.append("  * Latency numbers are typical p50 from public docs; actual latency depends on region/load.")
    out.append("")

    out.append(_hr("═"))
    return "\n".join(out)


def render_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=False)
