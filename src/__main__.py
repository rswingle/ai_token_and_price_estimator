"""CLI entry point: `python -m src "<project prompt>"`"""
from __future__ import annotations

import argparse
import json
import sys

from .cost_engine import compute_all, rank_by_cost
from .model_filter import filter_records
from .pricing_research import get_pricing
from .recommender import DEFAULT_WEIGHTS, recommend
from .report import build_human, build_json, render_json
from .token_estimator import estimate


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-token-price-estimator",
        description="Estimate LLM project token usage and rank stateless API providers by cost.",
    )
    p.add_argument(
        "prompt",
        nargs="?",
        help="Free-form project prompt. If omitted, read from stdin.",
    )
    p.add_argument(
        "--target-output",
        type=int,
        default=None,
        help="Override the heuristic and supply an exact target output token count.",
    )
    p.add_argument(
        "--multiplier",
        type=float,
        default=None,
        help="Override the output/input ratio (skips heuristic).",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Attempt a live web refresh of provider pricing (short timeout).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip disk cache; use the curated baseline directly.",
    )
    p.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Merge live prices from OpenRouter's public /api/v1/models (opt-in; falls back to baseline on any failure).",
    )
    p.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated models in the evaluation set.",
    )
    p.add_argument(
        "--format",
        choices=("human", "json", "both"),
        default="both",
        help="Output format (default: both).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of recommendations to return (default: 3).",
    )
    p.add_argument(
        "--weight-cost", type=float, default=None, help="Override weight: cost efficiency (default 0.40)."
    )
    p.add_argument(
        "--weight-capability", type=float, default=None, help="Override weight: capability fit (default 0.30)."
    )
    p.add_argument(
        "--weight-context", type=float, default=None, help="Override weight: context window fit (default 0.20)."
    )
    p.add_argument(
        "--weight-latency", type=float, default=None, help="Override weight: latency (default 0.10)."
    )
    return p


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if sys.stdin and not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    print("Error: no project prompt provided. Pass it as an argument or via stdin.", file=sys.stderr)
    sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    prompt = _read_prompt(args)

    weights: dict[str, float] = {}
    for k, v in (
        ("cost", args.weight_cost),
        ("capability", args.weight_capability),
        ("context", args.weight_context),
        ("latency", args.weight_latency),
    ):
        if v is not None:
            weights[k] = v
    if weights:
        default_total = sum(DEFAULT_WEIGHTS.values())
        provided_total = sum(weights.values())
        if provided_total <= 0:
            print("Error: --weight-* values must sum to a positive number.", file=sys.stderr)
            return 2

    try:
        est = estimate(
            prompt,
            target_output_tokens=args.target_output,
            multiplier_override=args.multiplier,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    pricing = get_pricing(
        refresh=args.refresh,
        use_cache_if_available=not args.no_cache,
        use_openrouter=args.use_openrouter,
    )
    kept, excluded = filter_records(pricing["records"], include_deprecated=args.include_deprecated)

    if not kept:
        print("Error: no providers passed the stateless filter.", file=sys.stderr)
        return 3

    lines = compute_all(est, kept)
    ranked = rank_by_cost(lines)
    recs = recommend(ranked, est.task_kind, weights=weights or None, top_n=args.top)

    j = build_json(prompt, est, pricing, kept, excluded, ranked, recs)

    if args.format in ("human", "both"):
        print(build_human(prompt, est, pricing, excluded, ranked, recs))
    if args.format in ("json", "both"):
        if args.format == "both":
            print()
            print("=== JSON OUTPUT ===")
        print(render_json(j))
    return 0


if __name__ == "__main__":
    sys.exit(main())
