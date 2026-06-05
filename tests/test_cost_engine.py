"""Unit tests for cost_engine module."""
from src.cost_engine import compute_all, compute_cost_line, rank_by_cost
from src.token_estimator import estimate


def _record(**overrides):
    base = {
        "provider": "Test",
        "model": "test-model",
        "input_cost_per_1m": 2.0,
        "output_cost_per_1m": 8.0,
        "context_window": 128_000,
        "source_url": "https://example.com/pricing",
        "fetched_at": "2026-01-01T00:00:00Z",
        "is_stateless": True,
        "deprecated": False,
        "capability_tags": ["reasoning", "coding"],
        "reasoning_strength": 0.8,
        "coding_strength": 0.9,
        "creativity_strength": 0.7,
        "latency_p50_ms": 500,
    }
    base.update(overrides)
    return base


class TestCostMath:
    def test_formula_exact(self):
        e = estimate("hello", multiplier_override=1.0)
        rec = _record(input_cost_per_1m=1.0, output_cost_per_1m=2.0)
        line = compute_cost_line(e, rec)
        # input_cost = input_tokens * 1/1_000_000
        assert line.input_cost == pytest_round(e.input_tokens * 1.0 / 1_000_000)
        # output_cost = output_tokens * 2/1_000_000
        assert line.output_cost == pytest_round(e.output_tokens_estimated * 2.0 / 1_000_000)

    def test_total_equals_input_plus_output(self):
        e = estimate("Implement X in Python")
        line = compute_cost_line(e, _record())
        assert abs(line.total_cost - (line.input_cost + line.output_cost)) < 1e-9

    def test_zero_pricing(self):
        e = estimate("hello")
        rec = _record(input_cost_per_1m=0.0, output_cost_per_1m=0.0)
        line = compute_cost_line(e, rec)
        assert line.total_cost == 0.0


class TestContextWindow:
    def test_fits_when_under(self):
        e = estimate("hi")
        line = compute_cost_line(e, _record(context_window=100_000))
        assert line.fits_context_window is True

    def test_does_not_fit_when_over(self):
        big_prompt = "word " * 50_000
        e = estimate(big_prompt)
        line = compute_cost_line(e, _record(context_window=1000))
        assert line.fits_context_window is False

    def test_zero_context_window_means_fits(self):
        e = estimate("hi")
        line = compute_cost_line(e, _record(context_window=0))
        assert line.fits_context_window is True


class TestRank:
    def test_rank_ascending(self):
        e = estimate("Implement X in Python")
        records = [
            _record(provider="A", model="a", input_cost_per_1m=10, output_cost_per_1m=20),
            _record(provider="B", model="b", input_cost_per_1m=1, output_cost_per_1m=2),
            _record(provider="C", model="c", input_cost_per_1m=5, output_cost_per_1m=10),
        ]
        lines = compute_all(e, records)
        ranked = rank_by_cost(lines)
        assert [l.provider for l in ranked] == ["B", "C", "A"]

    def test_rank_empty(self):
        assert rank_by_cost([]) == []

    def test_compute_all_returns_one_per_record(self):
        e = estimate("Implement X in Python")
        records = [_record(provider=f"P{i}", model=f"m{i}") for i in range(5)]
        assert len(compute_all(e, records)) == 5


def pytest_round(x: float) -> float:
    return round(x, 6)
