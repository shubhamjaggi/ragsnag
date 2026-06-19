"""Tests for Pydantic model validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragsnag._models import (
    Chunk,
    EvaluationResult,
    GeneratorOutput,
    LoopIteration,
    LoopResult,
    ReformulationOutput,
    ReformulationStrategy,
    StopReason,
)

from tests.conftest import make_chunk, make_evaluation, make_iteration


# ── Chunk ──────────────────────────────────────────────────────────────────────

def test_chunk_valid() -> None:
    chunk = Chunk(content="text", source="doc.pdf", score=0.85)
    assert chunk.content == "text"
    assert chunk.score == 0.85


def test_chunk_score_boundary_zero() -> None:
    chunk = Chunk(content="text", source="doc.pdf", score=0.0)
    assert chunk.score == 0.0


def test_chunk_score_boundary_one() -> None:
    chunk = Chunk(content="text", source="doc.pdf", score=1.0)
    assert chunk.score == 1.0


def test_chunk_score_above_one_raises() -> None:
    with pytest.raises(ValidationError):
        Chunk(content="text", source="doc.pdf", score=1.1)


def test_chunk_score_below_zero_raises() -> None:
    with pytest.raises(ValidationError):
        Chunk(content="text", source="doc.pdf", score=-0.1)


def test_chunk_default_metadata_is_empty_dict() -> None:
    chunk = Chunk(content="text", source="doc.pdf", score=0.5)
    assert chunk.metadata == {}


def test_chunk_metadata_stored() -> None:
    chunk = Chunk(content="text", source="doc.pdf", score=0.5, metadata={"page": 3})
    assert chunk.metadata["page"] == 3


# ── EvaluationResult ──────────────────────────────────────────────────────────

def test_evaluation_result_valid() -> None:
    result = EvaluationResult(is_grounded=True, is_complete=True, score=0.9, reason="good")
    assert result.score == 0.9


def test_evaluation_result_score_zero() -> None:
    result = EvaluationResult(is_grounded=False, is_complete=False, score=0.0, reason="bad")
    assert result.score == 0.0


def test_evaluation_result_score_one() -> None:
    result = EvaluationResult(is_grounded=True, is_complete=True, score=1.0, reason="perfect")
    assert result.score == 1.0


def test_evaluation_result_score_above_one_raises() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(is_grounded=True, is_complete=True, score=1.5, reason="r")


def test_evaluation_result_score_below_zero_raises() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(is_grounded=False, is_complete=False, score=-0.1, reason="r")


# ── GeneratorOutput ───────────────────────────────────────────────────────────

def test_generator_output_valid() -> None:
    out = GeneratorOutput(answer="answer", confidence=0.8)
    assert out.answer == "answer"
    assert out.confidence == 0.8


def test_generator_output_confidence_boundary() -> None:
    GeneratorOutput(answer="a", confidence=0.0)
    GeneratorOutput(answer="a", confidence=1.0)


def test_generator_output_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        GeneratorOutput(answer="a", confidence=1.1)


# ── ReformulationOutput ───────────────────────────────────────────────────────

def test_reformulation_output_valid() -> None:
    out = ReformulationOutput(
        queries=["query one"],
        strategy=ReformulationStrategy.EXPAND,
        reasoning="test",
    )
    assert len(out.queries) == 1


def test_reformulation_output_empty_queries_raises() -> None:
    with pytest.raises(ValidationError):
        ReformulationOutput(
            queries=[],
            strategy=ReformulationStrategy.EXPAND,
            reasoning="test",
        )


def test_reformulation_output_multiple_queries() -> None:
    out = ReformulationOutput(
        queries=["q1", "q2", "q3"],
        strategy=ReformulationStrategy.DECOMPOSE,
        reasoning="decomposing",
    )
    assert len(out.queries) == 3


# ── LoopResult ────────────────────────────────────────────────────────────────

def test_loop_result_best_iteration_returns_highest_score() -> None:
    from ragsnag._models import LoopResult
    iterations = [
        make_iteration(iteration=1, score=0.4),
        make_iteration(iteration=2, score=0.9),
        make_iteration(iteration=3, score=0.6),
    ]
    result = LoopResult(
        answer="answer",
        confidence=0.9,
        iterations=3,
        stop_reason=StopReason.MAX_ITERATIONS,
        trace=iterations,
    )
    assert result.best_iteration.evaluation.score == 0.9
    assert result.best_iteration.iteration == 2


def test_loop_result_best_iteration_single_trace() -> None:
    from ragsnag._models import LoopResult
    result = LoopResult(
        answer="answer",
        confidence=0.8,
        iterations=1,
        stop_reason=StopReason.CONVERGED,
        trace=[make_iteration(iteration=1, score=0.85)],
    )
    assert result.best_iteration.iteration == 1


# ── StopReason ────────────────────────────────────────────────────────────────

def test_stop_reason_values_exist() -> None:
    assert StopReason.CONVERGED == "converged"
    assert StopReason.MAX_ITERATIONS == "max_iterations"
    assert StopReason.HUMAN_APPROVED == "human_approved"
    assert StopReason.ERROR == "error"


# ── ReformulationStrategy ─────────────────────────────────────────────────────

def test_reformulation_strategy_values_exist() -> None:
    assert ReformulationStrategy.EXPAND == "expand"
    assert ReformulationStrategy.NARROW == "narrow"
    assert ReformulationStrategy.DECOMPOSE == "decompose"
    assert ReformulationStrategy.STEP_BACK == "step_back"
    assert ReformulationStrategy.HYDE == "hyde"
    assert ReformulationStrategy.PERSPECTIVE_SHIFT == "perspective_shift"
