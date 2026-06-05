"""Cost calculation engine.

Pure functions: given a token estimate and a list of model pricing records,
compute the per-model project cost using the standard formula and rank
the results.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .token_estimator import TokenEstimate

ONE_MILLION = 1_000_000


@dataclass
class CostLine:
    provider: str
    model: str
    context_window: int
    input_tokens: int
    output_tokens: int
    reasoning_buffer_tokens: int
    input_cost_per_1m: float
    output_cost_per_1m: float
    input_cost: float
    output_cost: float
    total_cost: float
    fits_context_window: bool
    source_url: str
    fetched_at: str
    capability_tags: list[str]
    reasoning_strength: float
    coding_strength: float
    creativity_strength: float
    latency_p50_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_cost_line(estimate: TokenEstimate, record: dict[str, Any]) -> CostLine:
    inp_price = float(record["input_cost_per_1m"])
    out_price = float(record["output_cost_per_1m"])
    context = int(record.get("context_window", 0))
    total_input = estimate.input_tokens
    fits = total_input <= context if context > 0 else True

    input_cost = total_input * (inp_price / ONE_MILLION)
    output_cost = estimate.output_tokens_estimated * (out_price / ONE_MILLION)
    total_cost = input_cost + output_cost

    return CostLine(
        provider=record.get("provider", ""),
        model=record.get("model", ""),
        context_window=context,
        input_tokens=total_input,
        output_tokens=estimate.output_tokens_estimated,
        reasoning_buffer_tokens=estimate.reasoning_buffer_tokens,
        input_cost_per_1m=inp_price,
        output_cost_per_1m=out_price,
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        total_cost=round(total_cost, 6),
        fits_context_window=fits,
        source_url=record.get("source_url", ""),
        fetched_at=record.get("fetched_at", ""),
        capability_tags=list(record.get("capability_tags", [])),
        reasoning_strength=float(record.get("reasoning_strength", 0.5)),
        coding_strength=float(record.get("coding_strength", 0.5)),
        creativity_strength=float(record.get("creativity_strength", 0.5)),
        latency_p50_ms=record.get("latency_p50_ms"),
    )


def compute_all(estimate: TokenEstimate, records: list[dict[str, Any]]) -> list[CostLine]:
    return [compute_cost_line(estimate, r) for r in records]


def rank_by_cost(lines: list[CostLine]) -> list[CostLine]:
    return sorted(lines, key=lambda x: x.total_cost)
