"""Weighted recommendation engine.

Default weights (per spec):
  cost efficiency:          40%
  capability fit:           30%
  context window fit:       20%
  latency:                  10%

`capability fit` blends reasoning/coding/creativity per task kind.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .cost_engine import CostLine
from .token_estimator import TaskKind

DEFAULT_WEIGHTS = {
    "cost": 0.40,
    "capability": 0.30,
    "context": 0.20,
    "latency": 0.10,
}

TASK_CAPABILITY_BLEND: dict[TaskKind, tuple[float, float, float]] = {
    "qa":       (0.40, 0.20, 0.40),
    "analysis": (0.55, 0.20, 0.25),
    "general":  (0.40, 0.30, 0.30),
    "longform": (0.20, 0.10, 0.70),
    "code":     (0.25, 0.65, 0.10),
    "agentic":  (0.55, 0.30, 0.15),
}


@dataclass
class ScoredLine:
    cost_line: CostLine
    cost_score: float
    capability_score: float
    context_score: float
    latency_score: float
    weighted_score: float
    rank: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_line"] = self.cost_line.to_dict()
        return d


def _safe_min_max(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return 0.0, 1.0
    return lo, hi


def _normalize(value: float, lo: float, hi: float, *, invert: bool = False) -> float:
    if hi - lo < 1e-9:
        return 1.0
    score = (value - lo) / (hi - lo)
    if invert:
        score = 1.0 - score
    return max(0.0, min(1.0, score))


def _capability_score(line: CostLine, task_kind: TaskKind) -> float:
    r_w, c_w, cr_w = TASK_CAPABILITY_BLEND[task_kind]
    blended = (
        line.reasoning_strength * r_w
        + line.coding_strength * c_w
        + line.creativity_strength * cr_w
    )
    return max(0.0, min(1.0, blended))


def _context_score(line: CostLine) -> float:
    if line.context_window <= 0:
        return 0.5
    if not line.fits_context_window:
        return 0.0
    utilization = line.input_tokens / line.context_window
    return max(0.2, 1.0 - utilization)


def _latency_score(line: CostLine) -> float:
    if line.latency_p50_ms is None or line.latency_p50_ms <= 0:
        return 0.5
    if line.latency_p50_ms <= 200:
        return 1.0
    if line.latency_p50_ms >= 3000:
        return 0.0
    return 1.0 - ((line.latency_p50_ms - 200) / (3000 - 200))


def _build_reason(line: CostLine, task_kind: TaskKind, weighted: float) -> str:
    bits: list[str] = []
    bits.append(f"task={task_kind}")
    bits.append(f"weighted_score={weighted:.3f}")
    if line.fits_context_window:
        bits.append(f"ctx={line.input_tokens}/{line.context_window}")
    else:
        bits.append(f"CTX_MISMATCH ({line.input_tokens}>{line.context_window})")
    if line.latency_p50_ms is not None:
        bits.append(f"p50={line.latency_p50_ms}ms")
    return " | ".join(bits)


def recommend(
    lines: list[CostLine],
    task_kind: TaskKind,
    weights: dict[str, float] | None = None,
    top_n: int = 3,
) -> list[ScoredLine]:
    """Score and rank CostLines; return the top_n recommendations.

    Lines that don't fit the context window are penalized (context_score=0)
    but not excluded unless they are the only option.
    """
    if not lines:
        return []
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    total_w = sum(w.values()) or 1.0
    w = {k: v / total_w for k, v in w.items()}

    costs = [ln.total_cost for ln in lines]
    cost_lo, cost_hi = _safe_min_max(costs)

    scored: list[ScoredLine] = []
    for ln in lines:
        cost_score = _normalize(ln.total_cost, cost_lo, cost_hi, invert=True)
        cap_score = _capability_score(ln, task_kind)
        ctx_score = _context_score(ln)
        lat_score = _latency_score(ln)

        weighted = (
            cost_score * w["cost"]
            + cap_score * w["capability"]
            + ctx_score * w["context"]
            + lat_score * w["latency"]
        )
        scored.append(ScoredLine(
            cost_line=ln,
            cost_score=round(cost_score, 4),
            capability_score=round(cap_score, 4),
            context_score=round(ctx_score, 4),
            latency_score=round(lat_score, 4),
            weighted_score=round(weighted, 4),
            rank=0,
            reason="",
        ))

    scored.sort(key=lambda s: s.weighted_score, reverse=True)
    for i, s in enumerate(scored, 1):
        s.rank = i
        s.reason = _build_reason(s.cost_line, task_kind, s.weighted_score)
    return scored[:top_n]
