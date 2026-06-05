"""Unit tests for pricing_research module."""
from unittest.mock import patch

from src.pricing_research import (
    _MODEL_PRICE_WINDOW,
    _refresh_one,
    _scrape_prices,
    _prices_in_window,
    _normalize_url,
)


class TestPricesInWindow:
    def test_returns_values_within_window(self):
        text = "x" * 200 + "gemini-2.5-pro costs $1.25 / 1M input and $10 / 1M output"
        anchor = text.find("gemini-2.5-pro")
        vals = _prices_in_window(text, anchor)
        assert vals == [1.25, 10.0]

    def test_excludes_values_outside_window(self):
        near = "gemini-2.5-pro $1.25 / 1M $10 / 1M"
        far = "x" * (_MODEL_PRICE_WINDOW + 200) + "$999 / 1M"
        text = near + far
        anchor = text.find("gemini-2.5-pro")
        vals = _prices_in_window(text, anchor)
        assert 999.0 not in vals
        assert 1.25 in vals and 10.0 in vals

    def test_returns_empty_for_negative_anchor(self):
        assert _prices_in_window("anything", -1) == []

    def test_excludes_out_of_sanity_range(self):
        text = "model-name $0.00001 / 1M" + " x" * 100 + " $2000 / 1M"
        anchor = 0
        vals = _prices_in_window(text, anchor)
        assert vals == []

    def test_handles_window_spanning_text_endpoints(self):
        anchor = 5
        text = "x" * anchor + "tiny"
        assert _prices_in_window(text, anchor) == []


class TestScrapePrices:
    def test_no_model_keyword_returns_empty(self):
        assert _scrape_prices("https://x", model_keyword=None) == {}

    def test_empty_text_returns_empty(self):
        with patch("src.pricing_research._fetch_page_text", return_value=""):
            assert _scrape_prices("https://x", model_keyword="gpt-4o") == {}

    def test_model_not_in_page_returns_empty(self):
        with patch(
            "src.pricing_research._fetch_page_text",
            return_value="completely unrelated content with no models listed",
        ):
            assert _scrape_prices("https://x", model_keyword="gpt-4o") == {}

    def test_one_price_in_window_returns_empty(self):
        text = "gpt-4o costs $2.50 / 1M for input"
        with patch("src.pricing_research._fetch_page_text", return_value=text):
            assert _scrape_prices("https://x", model_keyword="gpt-4o") == {}

    def test_two_prices_in_window_returns_pair(self):
        text = "gpt-4o costs $2.50 / 1M input and $10.00 / 1M output"
        with patch("src.pricing_research._fetch_page_text", return_value=text):
            result = _scrape_prices("https://x", model_keyword="gpt-4o")
        assert result == {"gpt-4o": (2.50, 10.00)}

    def test_does_not_pick_prices_from_other_models(self):
        text = (
            "gemini-2.5-pro costs $1.25 / 1M input and $10 / 1M output. "
            "x" * 800 + " "
            "gemini-2.5-flash costs $0.30 / 1M input and $2.50 / 1M output"
        )
        with patch("src.pricing_research._fetch_page_text", return_value=text):
            result = _scrape_prices("https://x", model_keyword="gemini-2.5-pro")
        assert result == {"gemini-2.5-pro": (1.25, 10.0)}
        assert 0.30 not in result["gemini-2.5-pro"]
        assert 2.50 not in result["gemini-2.5-pro"]


class TestRefreshOneConservative:
    def _entry(self, **overrides):
        base = {
            "provider": "Test",
            "model": "test-model",
            "input_cost_per_1m": 1.0,
            "output_cost_per_1m": 2.0,
            "context_window": 128000,
            "source_url": "https://old.example.com",
            "fetched_at": "2025-01-01T00:00:00Z",
            "is_stateless": True,
            "deprecated": False,
        }
        base.update(overrides)
        return base

    def test_no_per_model_match_returns_entry_unchanged(self):
        entry = self._entry()
        with patch("src.pricing_research._scrape_prices", return_value={}):
            result = _refresh_one(entry, "https://new.example.com")
        assert result is entry or result == entry

    def test_per_model_match_updates_prices(self):
        entry = self._entry(input_cost_per_1m=1.0, output_cost_per_1m=2.0)
        with patch(
            "src.pricing_research._scrape_prices",
            return_value={"test-model": (5.0, 15.0)},
        ):
            result = _refresh_one(entry, "https://new.example.com")
        assert result["input_cost_per_1m"] == 5.0
        assert result["output_cost_per_1m"] == 15.0
        assert result["source_url"] == _normalize_url("https://new.example.com")
        assert result["fetched_at"] != entry["fetched_at"]

    def test_no_change_when_extracted_matches_existing(self):
        entry = self._entry(input_cost_per_1m=5.0, output_cost_per_1m=15.0)
        with patch(
            "src.pricing_research._scrape_prices",
            return_value={"test-model": (5.0, 15.0)},
        ):
            result = _refresh_one(entry, "https://new.example.com")
        assert result["source_url"] == entry["source_url"]
        assert result["fetched_at"] == entry["fetched_at"]

    def test_rejects_out_of_sanity_prices(self):
        entry = self._entry()
        with patch(
            "src.pricing_research._scrape_prices",
            return_value={"test-model": (0.00001, 100)},
        ):
            result = _refresh_one(entry, "https://new.example.com")
        assert result["input_cost_per_1m"] == entry["input_cost_per_1m"]
        assert result["output_cost_per_1m"] == entry["output_cost_per_1m"]


class TestNormalizeUrl:
    def test_strips_query_and_fragment(self):
        assert _normalize_url("https://example.com/path?a=1#frag") == "https://example.com/path"

    def test_trailing_slash_removed(self):
        assert _normalize_url("https://example.com/path/") == "https://example.com/path"
