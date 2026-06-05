"""Unit tests for model_filter module."""
from src.model_filter import (
    filter_records,
    has_required_pricing,
    is_deprecated,
    is_session_based,
)


def _rec(**overrides):
    base = {
        "provider": "TestProvider",
        "model": "test-model",
        "input_cost_per_1m": 1.0,
        "output_cost_per_1m": 2.0,
        "context_window": 128_000,
        "source_url": "https://example.com",
        "fetched_at": "2026-01-01T00:00:00Z",
        "is_stateless": True,
        "deprecated": False,
        "capability_tags": [],
    }
    base.update(overrides)
    return base


class TestIsSessionBased:
    def test_chatgpt_plus(self):
        assert is_session_based(_rec(provider="OpenAI", model="ChatGPT Plus"))

    def test_claude_pro(self):
        assert is_session_based(_rec(provider="Anthropic", model="Claude Pro"))

    def test_perplexity_pro(self):
        assert is_session_based(_rec(provider="Perplexity", model="perplexity-pro"))

    def test_gemini_advanced(self):
        assert is_session_based(_rec(provider="Google", model="Gemini Advanced"))

    def test_copilot_pro(self):
        assert is_session_based(_rec(provider="Microsoft", model="Copilot Pro"))

    def test_poe_pro(self):
        assert is_session_based(_rec(provider="Poe", model="Poe Pro"))

    def test_stateless_models_pass(self):
        assert not is_session_based(_rec(model="gpt-4o"))
        assert not is_session_based(_rec(model="claude-sonnet-4-5"))
        assert not is_session_based(_rec(model="gemini-2.5-pro"))
        assert not is_session_based(_rec(model="deepseek-chat"))
        assert not is_session_based(_rec(model="llama-3.3-70b-versatile"))

    def test_explicit_flag(self):
        assert is_session_based(_rec(model="gpt-4o", is_stateless=False))


class TestIsDeprecated:
    def test_old_gpt_models(self):
        assert is_deprecated(_rec(model="gpt-3.5-turbo-0613"))
        assert is_deprecated(_rec(model="text-davinci-003"))

    def test_old_claude_models(self):
        assert is_deprecated(_rec(model="claude-2"))
        assert is_deprecated(_rec(model="claude-instant-1.2"))

    def test_current_models_pass(self):
        assert not is_deprecated(_rec(model="gpt-4o"))
        assert not is_deprecated(_rec(model="claude-sonnet-4-5"))
        assert not is_deprecated(_rec(model="gemini-2.5-pro"))

    def test_explicit_flag(self):
        assert is_deprecated(_rec(model="gpt-4o", deprecated=True))


class TestHasRequiredPricing:
    def test_valid(self):
        assert has_required_pricing(_rec())

    def test_negative(self):
        assert not has_required_pricing(_rec(input_cost_per_1m=-1))
        assert not has_required_pricing(_rec(output_cost_per_1m=-1))

    def test_zero_is_valid(self):
        assert has_required_pricing(_rec(input_cost_per_1m=0.0, output_cost_per_1m=0.0))

    def test_too_high(self):
        assert not has_required_pricing(_rec(input_cost_per_1m=5000))
        assert not has_required_pricing(_rec(output_cost_per_1m=5000))

    def test_missing(self):
        assert not has_required_pricing({"model": "x"})


class TestFilterRecords:
    def test_keeps_only_valid(self):
        records = [
            _rec(provider="OpenAI", model="gpt-4o"),
            _rec(provider="OpenAI", model="ChatGPT Plus"),
            _rec(provider="OpenAI", model="gpt-3.5-turbo-0613"),
            _rec(provider="Broken", model="bad", input_cost_per_1m=-1),
        ]
        kept, excluded = filter_records(records)
        assert len(kept) == 1
        assert kept[0]["model"] == "gpt-4o"
        assert len(excluded) == 3

    def test_include_deprecated(self):
        records = [_rec(provider="OpenAI", model="gpt-3.5-turbo-0613")]
        kept, excluded = filter_records(records, include_deprecated=True)
        assert len(kept) == 1
        assert len(excluded) == 0

    def test_excluded_reasons_listed(self):
        records = [_rec(provider="OpenAI", model="ChatGPT Plus")]
        _, excluded = filter_records(records)
        assert excluded[0]["reasons"]
        assert "session-based" in excluded[0]["reasons"][0]

    def test_empty_input(self):
        kept, excluded = filter_records([])
        assert kept == []
        assert excluded == []
