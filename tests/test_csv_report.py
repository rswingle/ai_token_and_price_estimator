"""Unit tests for the CSV builder."""
import csv
import io

from src.cost_engine import compute_all, rank_by_cost
from src.csv_report import COLUMNS, build_csv
from src.pricing_research import _baseline_snapshot
from src.recommender import recommend
from src.token_estimator import estimate


def _full_run(prompt="Implement a function in Python", top_n=3):
    est = estimate(prompt)
    records = _baseline_snapshot()
    lines = compute_all(est, records)
    ranked = rank_by_cost(lines)
    recs = recommend(ranked, est.task_kind, top_n=top_n)
    pricing_meta = {
        "source": "baseline",
        "stale": True,
        "fetched_at": "2026-06-05T00:00:00Z",
        "errors": [],
    }
    return prompt, est, pricing_meta, records, [], ranked, recs


def _parse(csv_text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(csv_text)))


class TestBuildCsv:
    def test_returns_string(self):
        text = build_csv(*_full_run())
        assert isinstance(text, str)
        assert text

    def test_columns_constant_is_ordered(self):
        assert COLUMNS[0] == "section"
        assert "section" in COLUMNS
        assert "prompt" in COLUMNS
        assert "total_cost_usd" in COLUMNS

    def test_contains_all_five_sections(self):
        rows = _parse(build_csv(*_full_run()))
        sections = {r["section"] for r in rows}
        assert sections == {
            "token_estimate",
            "recommendation",
            "pricing",
            "providers",
            "excluded",
        }

    def test_every_row_has_section(self):
        rows = _parse(build_csv(*_full_run()))
        assert rows
        assert all(r["section"] for r in rows)

    def test_token_estimate_row_present(self):
        rows = _parse(build_csv(*_full_run(prompt="hello world")))
        te = [r for r in rows if r["section"] == "token_estimate"]
        assert len(te) == 1
        assert te[0]["prompt"] == "hello world"
        assert te[0]["task_kind"]
        assert int(te[0]["input_tokens"].replace(",", "")) >= 0
        assert int(te[0]["output_tokens"].replace(",", "")) >= 0
        assert int(te[0]["total_tokens"].replace(",", "")) >= 0

    def test_recommendation_rows_count_matches_top_n(self):
        rows = _parse(build_csv(*_full_run(top_n=3)))
        recs = [r for r in rows if r["section"] == "recommendation"]
        assert len(recs) == 3
        assert recs[0]["rank"] == "1"
        assert recs[0]["kind"] == "best"
        assert recs[1]["kind"] == "alternative"
        assert recs[0]["provider"]
        assert recs[0]["model"]

    def test_pricing_rows_match_baseline_count(self):
        rows = _parse(build_csv(*_full_run()))
        pricing = [r for r in rows if r["section"] == "pricing"]
        assert len(pricing) == 20

    def test_pricing_first_row_is_cheapest(self):
        rows = _parse(build_csv(*_full_run()))
        pricing = [r for r in rows if r["section"] == "pricing"]
        costs = [
            float(r["total_cost_usd"].strip("$").replace(",", ""))
            for r in pricing
        ]
        assert costs == sorted(costs)

    def test_pricing_uses_money_format(self):
        rows = _parse(build_csv(*_full_run()))
        pricing = [r for r in rows if r["section"] == "pricing"]
        for r in pricing:
            assert r["total_cost_usd"].startswith("$"), r["total_cost_usd"]
            assert r["input_usd_per_1m"].startswith("$")
            assert r["output_usd_per_1m"].startswith("$")

    def test_providers_rows_aggregate(self):
        rows = _parse(build_csv(*_full_run()))
        providers = [r for r in rows if r["section"] == "providers"]
        assert len(providers) >= 1
        for r in providers:
            assert int(r["model_count"]) >= 1
            assert r["provider"]
            assert r["total_cost_usd"].startswith("$")

    def test_excluded_placeholder_when_empty(self):
        prompt, est, pricing_meta, kept, _, ranked, recs = _full_run()
        text = build_csv(prompt, est, pricing_meta, kept, [], ranked, recs)
        rows = _parse(text)
        excl = [r for r in rows if r["section"] == "excluded"]
        assert len(excl) == 1
        assert "No records were excluded" in excl[0]["note"]

    def test_excluded_records_listed_with_reason(self):
        prompt, est, pricing_meta, kept, _, ranked, recs = _full_run()
        excluded = [
            {"provider": "Bad", "model": "x", "reasons": ["deprecated model"]},
            {
                "provider": "Worse",
                "model": "y",
                "reasons": ["session-based or chat product"],
            },
        ]
        text = build_csv(prompt, est, pricing_meta, kept, excluded, ranked, recs)
        rows = _parse(text)
        excl = [r for r in rows if r["section"] == "excluded"]
        assert len(excl) == 2
        assert excl[0]["provider"] == "Bad"
        assert "deprecated" in excl[0]["reason"]
        assert excl[1]["provider"] == "Worse"
        assert "chat" in excl[1]["reason"]

    def test_score_uses_four_decimals(self):
        rows = _parse(build_csv(*_full_run()))
        recs = [r for r in rows if r["section"] == "recommendation"]
        for r in recs:
            assert r["weighted_score"]
            assert "." in r["weighted_score"]
            decimals = r["weighted_score"].split(".")[1]
            assert len(decimals) == 4

    def test_empty_recommendations_does_not_crash(self):
        prompt, est, pricing_meta, kept, _, ranked, _ = _full_run()
        text = build_csv(prompt, est, pricing_meta, kept, [], ranked, [])
        rows = _parse(text)
        recs = [r for r in rows if r["section"] == "recommendation"]
        assert recs == []

    def test_round_trip_through_file(self, tmp_path):
        text = build_csv(*_full_run())
        out = tmp_path / "report.csv"
        out.write_text(text, encoding="utf-8")
        loaded = list(csv.DictReader(out.open(encoding="utf-8")))
        assert any(r["section"] == "token_estimate" for r in loaded)
        assert any(r["section"] == "recommendation" for r in loaded)
        assert any(r["section"] == "pricing" for r in loaded)
        assert any(r["section"] == "providers" for r in loaded)
        assert any(r["section"] == "excluded" for r in loaded)
