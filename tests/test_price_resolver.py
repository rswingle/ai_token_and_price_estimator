"""Unit tests for the OpenRouter price resolver."""
from unittest.mock import patch, MagicMock

from src.price_resolver import (
    _OPENROUTER_MAP,
    _parse_price,
    apply_openrouter_prices,
    fetch_openrouter_prices,
)


def _baseline_record(provider, model, in_p, out_p):
    return {
        "provider": provider,
        "model": model,
        "input_cost_per_1m": in_p,
        "output_cost_per_1m": out_p,
        "context_window": 128000,
        "source_url": "https://old.example.com",
        "fetched_at": "2025-01-01T00:00:00Z",
        "is_stateless": True,
        "deprecated": False,
        "capability_tags": ["reasoning"],
        "reasoning_strength": 0.9,
        "coding_strength": 0.85,
        "creativity_strength": 0.8,
        "latency_p50_ms": 500,
    }


class TestParsePrice:
    def test_valid(self):
        assert _parse_price("0.0000025") == 2.5

    def test_none(self):
        assert _parse_price(None) is None

    def test_invalid_string(self):
        assert _parse_price("not-a-number") is None

    def test_out_of_range_low(self):
        assert _parse_price("0.00000000001") is None

    def test_out_of_range_high(self):
        assert _parse_price("2") is None


class TestFetchOpenrouterPrices:
    def _mock_response(self, payload, status=200):
        mock = MagicMock()
        mock.status_code = status
        mock.json.return_value = payload
        return mock

    def test_happy_path_converts_per_token_to_per_million(self):
        payload = {"data": [
            {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
        ]}
        with patch("src.price_resolver.requests.get", return_value=self._mock_response(payload)):
            result = fetch_openrouter_prices()
        assert result == {"openai/gpt-4o": (2.5, 10.0)}

    def test_skips_records_missing_prices(self):
        payload = {"data": [
            {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
            {"id": "vendor/broken", "pricing": {}},
            {"id": "vendor/half", "pricing": {"prompt": "0.000001"}},
        ]}
        with patch("src.price_resolver.requests.get", return_value=self._mock_response(payload)):
            result = fetch_openrouter_prices()
        assert "openai/gpt-4o" in result
        assert "vendor/broken" not in result
        assert "vendor/half" not in result

    def test_returns_none_on_non_200(self):
        with patch("src.price_resolver.requests.get", return_value=self._mock_response({}, status=503)):
            assert fetch_openrouter_prices() is None

    def test_returns_none_on_network_error(self):
        with patch("src.price_resolver.requests.get", side_effect=ConnectionError("nope")):
            assert fetch_openrouter_prices() is None

    def test_returns_none_on_invalid_json(self):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.side_effect = ValueError("bad json")
        with patch("src.price_resolver.requests.get", return_value=mock):
            assert fetch_openrouter_prices() is None

    def test_returns_none_when_no_usable_records(self):
        payload_bad = {"data": [
            {"id": "x/y", "pricing": {"prompt": "0.00000000001", "completion": "0.00000000001"}},
        ]}
        with patch("src.price_resolver.requests.get", return_value=self._mock_response(payload_bad)):
            assert fetch_openrouter_prices() is None


class TestApplyOpenrouterPrices:
    def test_updates_only_prices_for_mapped_records(self):
        baseline = [_baseline_record("OpenAI", "gpt-4o", 2.5, 10.0)]
        prices = {"openai/gpt-4o": (3.0, 12.0)}
        updated, changes, unmapped = apply_openrouter_prices(baseline, prices)
        assert changes == 1
        assert updated[0]["input_cost_per_1m"] == 3.0
        assert updated[0]["output_cost_per_1m"] == 12.0
        assert updated[0]["reasoning_strength"] == 0.9
        assert updated[0]["capability_tags"] == ["reasoning"]
        assert updated[0]["source_url"] == "https://openrouter.ai/models"
        assert unmapped == []

    def test_no_change_when_prices_match(self):
        baseline = [_baseline_record("OpenAI", "gpt-4o", 2.5, 10.0)]
        prices = {"openai/gpt-4o": (2.5, 10.0)}
        updated, changes, unmapped = apply_openrouter_prices(baseline, prices)
        assert changes == 0
        assert updated[0]["source_url"] == "https://old.example.com"
        assert unmapped == []

    def test_unmapped_record_left_unchanged_and_reported(self):
        baseline = [_baseline_record("MadeUp", "fake-model", 1.0, 2.0)]
        prices = {"openai/gpt-4o": (3.0, 12.0)}
        updated, changes, unmapped = apply_openrouter_prices(baseline, prices)
        assert changes == 0
        assert updated[0]["input_cost_per_1m"] == 1.0
        assert unmapped[0]["provider"] == "MadeUp"
        assert unmapped[0]["openrouter_id"] is None

    def test_mapped_but_missing_from_prices_left_unchanged(self):
        baseline = [_baseline_record("OpenAI", "gpt-4o", 2.5, 10.0)]
        prices: dict = {}
        updated, changes, unmapped = apply_openrouter_prices(baseline, prices)
        assert changes == 0
        assert updated[0]["input_cost_per_1m"] == 2.5
        assert unmapped[0]["openrouter_id"] == "openai/gpt-4o"

    def test_does_not_mutate_baseline(self):
        baseline = [_baseline_record("OpenAI", "gpt-4o", 2.5, 10.0)]
        prices = {"openai/gpt-4o": (3.0, 12.0)}
        apply_openrouter_prices(baseline, prices)
        assert baseline[0]["input_cost_per_1m"] == 2.5
        assert baseline[0]["output_cost_per_1m"] == 10.0


class TestOpenrouterMapCoverage:
    def test_all_baseline_providers_have_some_mappings(self):
        providers_in_map = {p for (p, _m) in _OPENROUTER_MAP}
        assert {"OpenAI", "Anthropic", "Google", "Mistral", "DeepSeek", "Groq"} <= providers_in_map

    def test_map_values_are_well_formed_openrouter_ids(self):
        for (provider, model), or_id in _OPENROUTER_MAP.items():
            assert "/" in or_id, f"{provider}/{model} -> {or_id} is not a vendor/model ID"
            assert " " not in or_id
