"""Token estimation for LLM projects.

Classifies the project prompt by intent, counts input tokens via tiktoken,
and estimates output + reasoning overhead using deterministic heuristics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Literal

import tiktoken

TaskKind = Literal["qa", "code", "agentic", "longform", "analysis", "general"]

TASK_MULTIPLIERS: dict[TaskKind, tuple[float, float]] = {
    "qa":       (0.5, 1.0),
    "analysis": (1.0, 2.0),
    "general":  (1.0, 2.0),
    "longform": (2.0, 5.0),
    "code":     (2.0, 4.0),
    "agentic":  (3.0, 6.0),
}

REASONING_BUFFER = {
    "qa":       0.10,
    "analysis": 0.20,
    "general":  0.20,
    "longform": 0.15,
    "code":     0.30,
    "agentic":  0.50,
}

SYSTEM_PROMPT_OVERHEAD = 80

_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


_CODE_PATTERNS = re.compile(
    r"\b(code|function|class|implement|build|debug|refactor|"
    r"api|endpoint|sql|script|algorithm|compile|test|unit test|"
    r"typescript|python|javascript|rust|go\b|java\b)\b",
    re.IGNORECASE,
)
_AGENTIC_PATTERNS = re.compile(
    r"\b(agent|agents|tool use|tools|multi[- ]step|orchestrate|"
    r"automat|workflow|pipeline|chain|router|planner|executor|"
    r"browse|scrape|fetch.*api|iterate|critique.*loop|react agent|"
    r"function[- ]calling|mcp|retrieval)\b",
    re.IGNORECASE,
)
_LONGFORM_PATTERNS = re.compile(
    r"\b(write|essay|article|blog|story|narrative|report|"
    r"whitepaper|newsletter|chapter|book|long[- ]form|"
    r"copywrite|email sequence|landing page)\b",
    re.IGNORECASE,
)
_QA_PATTERNS = re.compile(
    r"^(\s*(what|who|when|where|why|how|define|explain|describe|list|tell me)\b|"
    r"\?$)",
    re.IGNORECASE,
)
_ANALYSIS_PATTERNS = re.compile(
    r"(?<!\w)(analy[sz]e|summariz(?:e|ing|y)?|extract|classify|cluster|"
    r"categoriz|compare|evaluate|score|rank|"
    r"sentiment|topic|theme|pattern|trend|"
    r"review(?:\s+(?:code|document|paper|proposal))?)",
    re.IGNORECASE,
)


def classify(prompt: str) -> TaskKind:
    """Classify a project prompt by intent using deterministic keyword heuristics."""
    if _AGENTIC_PATTERNS.search(prompt):
        return "agentic"
    if _CODE_PATTERNS.search(prompt):
        return "code"
    if _LONGFORM_PATTERNS.search(prompt):
        return "longform"
    if _ANALYSIS_PATTERNS.search(prompt):
        return "analysis"
    if _QA_PATTERNS.search(prompt):
        return "qa"
    return "general"


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


@dataclass
class TokenEstimate:
    input_tokens: int
    output_tokens_estimated: int
    reasoning_buffer_tokens: int
    system_overhead_tokens: int
    total_tokens: int
    task_kind: TaskKind
    multiplier_used: float
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def estimate(
    project_prompt: str,
    target_output_tokens: int | None = None,
    multiplier_override: float | None = None,
) -> TokenEstimate:
    """Estimate total token usage for executing a project prompt.

    multiplier_override: force a specific output/input ratio (skip heuristic).
    target_output_tokens: if provided, bypass output estimation entirely.
    """
    if not project_prompt or not project_prompt.strip():
        raise ValueError("project_prompt must be a non-empty string")

    notes: list[str] = []
    kind = classify(project_prompt)
    notes.append(f"Classified task as: {kind}")

    input_tokens = count_tokens(project_prompt) + SYSTEM_PROMPT_OVERHEAD
    notes.append(f"Counted {input_tokens - SYSTEM_PROMPT_OVERHEAD} prompt tokens + {SYSTEM_PROMPT_OVERHEAD} system overhead")

    if multiplier_override is not None:
        if multiplier_override < 0:
            raise ValueError("multiplier_override must be >= 0")
        mult = multiplier_override
        notes.append(f"Using user-provided multiplier: {mult}x")
    else:
        lo, hi = TASK_MULTIPLIERS[kind]
        mid = (lo + hi) / 2.0
        mult = round(mid, 2)
        notes.append(f"Applied heuristic multiplier {mult}x (range {lo}x-{hi}x for {kind})")

    if target_output_tokens is not None:
        if target_output_tokens < 0:
            raise ValueError("target_output_tokens must be >= 0")
        output_estimate = int(target_output_tokens)
        notes.append(f"Using user-provided target output: {output_estimate} tokens")
    else:
        output_estimate = int(round(input_tokens * mult))
        notes.append(f"Heuristic output estimate: {input_tokens} * {mult} = {output_estimate}")

    reasoning_buffer = int(round(input_tokens * REASONING_BUFFER[kind]))
    notes.append(f"Reasoning/workflow buffer: {int(REASONING_BUFFER[kind] * 100)}% of input = {reasoning_buffer}")

    total = input_tokens + output_estimate + reasoning_buffer
    notes.append(f"Total: input {input_tokens} + output {output_estimate} + buffer {reasoning_buffer} = {total}")

    return TokenEstimate(
        input_tokens=input_tokens,
        output_tokens_estimated=output_estimate,
        reasoning_buffer_tokens=reasoning_buffer,
        system_overhead_tokens=SYSTEM_PROMPT_OVERHEAD,
        total_tokens=total,
        task_kind=kind,
        multiplier_used=mult,
        notes=notes,
    )
