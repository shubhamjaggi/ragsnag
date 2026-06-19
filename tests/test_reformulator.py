"""Tests for LLMReformulator and HeuristicReformulator."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ragsnag._models import ReformulationStrategy
from ragsnag._reformulator import (
    HeuristicReformulator,
    LLMReformulator,
    _select_strategy_heuristic,
)

from tests.conftest import make_evaluation, make_iteration


def make_llm_reformulator(response: str) -> tuple[LLMReformulator, MagicMock]:
    generate_fn = MagicMock(return_value=response)
    return LLMReformulator(generate_fn=generate_fn), generate_fn


def valid_response(
    strategy: str = "expand",
    queries: list[str] | None = None,
    reasoning: str = "test reasoning",
) -> str:
    return json.dumps({
        "strategy": strategy,
        "queries": queries or ["reformulated query"],
        "reasoning": reasoning,
    })


# ── LLMReformulator: parsing ───────────────────────────────────────────────────

def test_parses_valid_json() -> None:
    ref, _ = make_llm_reformulator(valid_response(strategy="narrow", queries=["specific query"]))
    result = ref.reformulate("original", [make_iteration()])
    assert result.strategy == ReformulationStrategy.NARROW
    assert result.queries == ["specific query"]


def test_parses_multiple_queries_for_decompose() -> None:
    ref, _ = make_llm_reformulator(
        valid_response(strategy="decompose", queries=["part one", "part two"])
    )
    result = ref.reformulate("original", [make_iteration()])
    assert len(result.queries) == 2
    assert "part one" in result.queries
    assert "part two" in result.queries


def test_parses_reasoning() -> None:
    ref, _ = make_llm_reformulator(valid_response(reasoning="Vocabulary mismatch detected."))
    result = ref.reformulate("original", [make_iteration()])
    assert result.reasoning == "Vocabulary mismatch detected."


def test_all_strategies_parse_correctly() -> None:
    for strategy in ReformulationStrategy:
        ref, _ = make_llm_reformulator(valid_response(strategy=strategy.value))
        result = ref.reformulate("original", [make_iteration()])
        assert result.strategy == strategy


# ── LLMReformulator: error handling ───────────────────────────────────────────

def test_fallback_on_invalid_json() -> None:
    ref, _ = make_llm_reformulator("not json")
    result = ref.reformulate("original query", [make_iteration(score=0.3, query="original query")])
    assert len(result.queries) >= 1
    assert result.strategy in ReformulationStrategy.__members__.values()


def test_fallback_returns_original_query_on_json_error() -> None:
    ref, _ = make_llm_reformulator("broken")
    result = ref.reformulate("original query", [make_iteration()])
    assert result.queries == ["original query"]


def test_fallback_on_invalid_strategy_value() -> None:
    response = json.dumps({"strategy": "nonexistent_strategy", "queries": ["q"], "reasoning": "r"})
    ref, _ = make_llm_reformulator(response)
    result = ref.reformulate("original query", [make_iteration()])
    assert result.strategy in ReformulationStrategy.__members__.values()


def test_fallback_uses_heuristic_strategy() -> None:
    history = [make_iteration(score=0.3)]
    history[0] = history[0].model_copy(
        update={"evaluation": make_evaluation(score=0.3, reason="too generic results returned")}
    )
    ref, _ = make_llm_reformulator("broken json")
    result = ref.reformulate("original", history)
    assert result.strategy == ReformulationStrategy.NARROW


# ── LLMReformulator: prompt ────────────────────────────────────────────────────

def test_prompt_contains_original_query() -> None:
    ref, generate_fn = make_llm_reformulator(valid_response())
    ref.reformulate("What is the late payment fee?", [make_iteration()])
    prompt = generate_fn.call_args[0][0]
    assert "What is the late payment fee?" in prompt


def test_prompt_contains_history_summary() -> None:
    ref, generate_fn = make_llm_reformulator(valid_response())
    history = [make_iteration(iteration=1, query="first attempt", score=0.4)]
    ref.reformulate("original", history)
    prompt = generate_fn.call_args[0][0]
    assert "first attempt" in prompt


def test_prompt_contains_all_iterations() -> None:
    ref, generate_fn = make_llm_reformulator(valid_response())
    history = [
        make_iteration(iteration=1, query="attempt one"),
        make_iteration(iteration=2, query="attempt two"),
    ]
    ref.reformulate("original", history)
    prompt = generate_fn.call_args[0][0]
    assert "attempt one" in prompt
    assert "attempt two" in prompt


def test_generate_fn_called_exactly_once() -> None:
    ref, generate_fn = make_llm_reformulator(valid_response())
    ref.reformulate("original", [make_iteration()])
    generate_fn.assert_called_once()


# ── HeuristicReformulator ──────────────────────────────────────────────────────

def make_history_with_reason(reason: str) -> list:
    return [make_iteration(score=0.3, query="test query")].__class__(
        [make_iteration(score=0.3)]
    )


def history_with_reason(reason: str) -> list:
    it = make_iteration(score=0.3)
    it = it.model_copy(update={"evaluation": make_evaluation(score=0.3, reason=reason)})
    return [it]


def test_heuristic_selects_perspective_shift_for_vocabulary() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate("q", history_with_reason("vocabulary mismatch between query and docs"))
    assert result.strategy == ReformulationStrategy.PERSPECTIVE_SHIFT


def test_heuristic_selects_perspective_shift_for_no_chunks() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate("q", history_with_reason("no chunk matched the query"))
    assert result.strategy == ReformulationStrategy.PERSPECTIVE_SHIFT


def test_heuristic_selects_narrow_for_too_generic() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate("q", history_with_reason("results are too generic"))
    assert result.strategy == ReformulationStrategy.NARROW


def test_heuristic_selects_narrow_for_vague() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate("q", history_with_reason("answer is too vague"))
    assert result.strategy == ReformulationStrategy.NARROW


def test_heuristic_selects_decompose_for_multi_part() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate("q", history_with_reason("multi-part question not addressed"))
    assert result.strategy == ReformulationStrategy.DECOMPOSE


def test_heuristic_selects_step_back_for_no_relevant() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate("q", history_with_reason("no relevant chunks retrieved"))
    assert result.strategy == ReformulationStrategy.STEP_BACK


def test_heuristic_defaults_to_expand_for_unknown_reason() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate("q", history_with_reason("something went wrong"))
    assert result.strategy == ReformulationStrategy.EXPAND


def test_heuristic_decompose_splits_on_and() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate(
        "pricing and billing cycle",
        history_with_reason("multi-part question"),
    )
    assert result.strategy == ReformulationStrategy.DECOMPOSE
    assert len(result.queries) == 2
    assert "pricing" in result.queries[0]
    assert "billing cycle" in result.queries[1]


def test_heuristic_decompose_no_and_keeps_single_query() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate(
        "what is the refund policy",
        history_with_reason("multi-part question"),
    )
    assert result.queries == ["what is the refund policy"]


def test_heuristic_reasoning_references_evaluation_reason() -> None:
    ref = HeuristicReformulator()
    reason = "vocabulary mismatch detected"
    result = ref.reformulate("q", history_with_reason(reason))
    assert reason[:50] in result.reasoning


def test_heuristic_always_returns_at_least_one_query() -> None:
    ref = HeuristicReformulator()
    result = ref.reformulate("q", history_with_reason("unknown issue"))
    assert len(result.queries) >= 1


# ── _select_strategy_heuristic ─────────────────────────────────────────────────

@pytest.mark.parametrize("reason,expected", [
    ("vocabulary mismatch", ReformulationStrategy.PERSPECTIVE_SHIFT),
    ("no chunk found", ReformulationStrategy.PERSPECTIVE_SHIFT),
    ("not retrieved", ReformulationStrategy.PERSPECTIVE_SHIFT),
    ("jargon in documents", ReformulationStrategy.PERSPECTIVE_SHIFT),
    ("too generic content", ReformulationStrategy.NARROW),
    ("too broad results", ReformulationStrategy.NARROW),
    ("surface level only", ReformulationStrategy.NARROW),
    ("multi-part question", ReformulationStrategy.DECOMPOSE),
    ("missing second part", ReformulationStrategy.DECOMPOSE),
    ("no relevant results", ReformulationStrategy.STEP_BACK),
    ("completely off topic", ReformulationStrategy.STEP_BACK),
    ("unknown reason xyz", ReformulationStrategy.EXPAND),
])
def test_select_strategy_heuristic(reason: str, expected: ReformulationStrategy) -> None:
    assert _select_strategy_heuristic(reason) == expected
