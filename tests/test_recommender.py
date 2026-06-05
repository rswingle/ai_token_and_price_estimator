"""Unit tests for recommender module."""
from src.cost_engine import compute_all
from src.recommender import DEFAULT_WEIGHTS, recommend
from src.token_estimator import estimate


def _rec(**overrides):
    base = {
        "provider": "Test",
        "model": "test",
        "input_cost_per_1m": 2.0,
        "output_cost_per_1m": 8.0,
        "context_window": 128_000,
        "source_url": "https://x",
        "fetched_at": "2026-01-01",
        "is_stateless": True,
        "deprecated": False,
        "capability_tags": [],
        "reasoning_strength": 0.8,
        "coding_strength": 0.8,
        "creativity_strength": 0.8,
        "latency_p50_ms": 500,
    }
    base.update(overrides)
    return base


class TestDefaults:
    def test_default_weights_sum_to_one(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


class TestRecommend:
    def test_empty_returns_empty(self):
        assert recommend([], "general", top_n=3) == []

    def test_returns_top_n(self):
        e = estimate("Implement a function in Python")
        records = [_rec(provider=f"P{i}", model=f"m{i}") for i in range(5)]
        lines = compute_all(e, records)
        recs = recommend(lines, "code", top_n=3)
        assert len(recs) == 3

    def test_ranks_sequential(self):
        e = estimate("Implement a function in Python")
        records = [_rec(provider=f"P{i}", model=f"m{i}") for i in range(4)]
        lines = compute_all(e, records)
        recs = recommend(lines, "code", top_n=4)
        assert [r.rank for r in recs] == [1, 2, 3, 4]

    def test_top_is_best_weighted(self):
        e = estimate("Implement a function in Python")
        cheap = _rec(provider="Cheap", model="c", input_cost_per_1m=0.1, output_cost_per_1m=0.1)
        expensive_strong = _rec(provider="Strong", model="s",
                                input_cost_per_1m=10, output_cost_per_1m=20,
                                reasoning_strength=0.99, coding_strength=0.99, creativity_strength=0.99)
        lines = compute_all(e, [cheap, expensive_strong])
        recs = recommend(lines, "code", top_n=2)
        assert recs[0].weighted_score >= recs[1].weighted_score

    def test_weights_override(self):
        e = estimate("Implement a function in Python")
        # pure cost-priority should pick cheapest
        cheap = _rec(provider="Cheap", model="c", input_cost_per_1m=0.1, output_cost_per_1m=0.1)
        costly = _rec(provider="Costly", model="x", input_cost_per_1m=10, output_cost_per_1m=20,
                      reasoning_strength=0.99, coding_strength=0.99, creativity_strength=0.99)
        lines = compute_all(e, [cheap, costly])
        recs = recommend(lines, "code", weights={"cost": 1.0, "capability": 0.0, "context": 0.0, "latency": 0.0})
        assert recs[0].cost_line.provider == "Cheap"

    def test_context_overflow_penalized(self):
        e = estimate("Implement a function in Python")
        ok = _rec(provider="Ok", model="o", context_window=200_000)
        overflow = _rec(provider="Over", model="ov", context_window=10)
        lines = compute_all(e, [ok, overflow])
        recs = recommend(lines, "code", top_n=2)
        assert recs[0].cost_line.provider == "Ok"
        assert recs[0].context_score > recs[1].context_score

    def test_reason_includes_task_kind(self):
        e = estimate("Build an agent that uses tools")
        lines = compute_all(e, [_rec()])
        recs = recommend(lines, "agentic", top_n=1)
        assert "task=agentic" in recs[0].reason
