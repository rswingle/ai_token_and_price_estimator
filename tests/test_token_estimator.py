"""Unit tests for token_estimator module."""
import pytest

from src.token_estimator import (
    TASK_MULTIPLIERS,
    classify,
    count_tokens,
    estimate,
)


class TestClassify:
    def test_qa_prompts(self):
        assert classify("What is the capital of France?") == "qa"
        assert classify("Explain quantum entanglement in simple terms") == "qa"

    def test_code_prompts(self):
        assert classify("Implement a function to compute fibonacci in Python") == "code"
        assert classify("Write a SQL query joining three tables") == "code"
        assert classify("Build a REST API with FastAPI") == "code"

    def test_agentic_prompts(self):
        assert classify("Build an agent that browses the web and writes a report") == "agentic"
        assert classify("Create a multi-step tool-using pipeline for data analysis") == "agentic"

    def test_longform_prompts(self):
        assert classify("Write a 2000-word essay on climate change") == "longform"
        assert classify("Draft a whitepaper on transformer architectures") == "longform"

    def test_analysis_prompts(self):
        assert classify("Analyze the sentiment of these customer reviews") == "analysis"
        assert classify("Summarize the key trends in the document") == "analysis"

    def test_general_fallback(self):
        assert classify("Help me think through a tricky problem at work") == "general"


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        n = count_tokens("Hello, world!")
        assert n > 0
        assert n < 20

    def test_deterministic(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert count_tokens(text) == count_tokens(text)


class TestEstimate:
    def test_basic_code_estimate(self):
        e = estimate("Implement fibonacci in Python")
        assert e.task_kind == "code"
        assert e.input_tokens > 80
        assert e.output_tokens_estimated > e.input_tokens
        assert e.total_tokens == e.input_tokens + e.output_tokens_estimated + e.reasoning_buffer_tokens

    def test_target_output_override(self):
        e = estimate("anything", target_output_tokens=2000)
        assert e.output_tokens_estimated == 2000
        assert e.notes and any("user-provided target" in n for n in e.notes)

    def test_multiplier_override(self):
        e = estimate("hello world", multiplier_override=5.0)
        assert e.multiplier_used == 5.0
        assert e.notes and any("user-provided multiplier" in n for n in e.notes)

    def test_deterministic_same_input(self):
        a = estimate("Implement fibonacci in Python")
        b = estimate("Implement fibonacci in Python")
        assert a.total_tokens == b.total_tokens
        assert a.output_tokens_estimated == b.output_tokens_estimated

    def test_different_tasks_different_outputs(self):
        qa = estimate("What is the capital of France?")
        code = estimate("Implement a Python function")
        longform = estimate("Write an essay on climate change")
        agentic = estimate("Build an agent that browses the web and writes a report")
        for e in (qa, code, longform, agentic):
            assert e.total_tokens > 0
        assert qa.multiplier_used < code.multiplier_used
        assert code.multiplier_used < agentic.multiplier_used
        assert qa.task_kind == "qa"
        assert code.task_kind == "code"
        assert longform.task_kind == "longform"
        assert agentic.task_kind == "agentic"

    def test_multipliers_per_task_match_spec(self):
        # Spec rules:
        # simple Q&A -> 1x
        # code generation -> 2-4x
        # agentic -> 3-6x
        # long-form -> 2-5x
        assert TASK_MULTIPLIERS["qa"] == (0.5, 1.0)
        assert TASK_MULTIPLIERS["code"] == (2.0, 4.0)
        assert TASK_MULTIPLIERS["agentic"] == (3.0, 6.0)
        assert TASK_MULTIPLIERS["longform"] == (2.0, 5.0)

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError):
            estimate("")

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError):
            estimate("hello", multiplier_override=-1.0)

    def test_negative_target_raises(self):
        with pytest.raises(ValueError):
            estimate("hello", target_output_tokens=-100)
