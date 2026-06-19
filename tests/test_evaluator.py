"""Tests for LLMEvaluator."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ragsnag._evaluator import LLMEvaluator
from ragsnag._models import Chunk

from tests.conftest import make_chunk


def make_evaluator(response: str) -> tuple[LLMEvaluator, MagicMock]:
    generate_fn = MagicMock(return_value=response)
    return LLMEvaluator(generate_fn=generate_fn), generate_fn


def valid_response(
    is_grounded: bool = True,
    is_complete: bool = True,
    score: float = 0.9,
    reason: str = "Answer is fully supported.",
) -> str:
    return json.dumps({
        "is_grounded": is_grounded,
        "is_complete": is_complete,
        "score": score,
        "reason": reason,
    })


# ── Parsing ───────────────────────────────────────────────────────────────────

def test_parses_valid_json_correctly() -> None:
    evaluator, _ = make_evaluator(valid_response(is_grounded=True, is_complete=True, score=0.92))
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert result.is_grounded is True
    assert result.is_complete is True
    assert result.score == 0.92


def test_parses_grounded_false() -> None:
    evaluator, _ = make_evaluator(valid_response(is_grounded=False, score=0.2))
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert result.is_grounded is False


def test_parses_complete_false() -> None:
    evaluator, _ = make_evaluator(valid_response(is_complete=False, score=0.4))
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert result.is_complete is False


def test_parses_reason_string() -> None:
    evaluator, _ = make_evaluator(valid_response(reason="Missing the international part."))
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert result.reason == "Missing the international part."


def test_handles_json_with_surrounding_whitespace() -> None:
    response = "   " + valid_response() + "\n"
    evaluator, _ = make_evaluator(response)
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert result.score > 0.0


# ── Error handling ────────────────────────────────────────────────────────────

def test_returns_zero_score_on_invalid_json() -> None:
    evaluator, _ = make_evaluator("not json at all")
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert result.score == 0.0
    assert result.is_grounded is False
    assert result.is_complete is False


def test_returns_zero_score_on_missing_key() -> None:
    response = json.dumps({"is_grounded": True, "score": 0.9})  # missing is_complete, reason
    evaluator, _ = make_evaluator(response)
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert result.score == 0.0


def test_error_reason_contains_raw_response_snippet() -> None:
    evaluator, _ = make_evaluator("UNPARSEABLE OUTPUT XYZ")
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert "UNPARSEABLE OUTPUT XYZ" in result.reason


def test_returns_zero_score_on_empty_response() -> None:
    evaluator, _ = make_evaluator("")
    result = evaluator.evaluate("q", [make_chunk()], "answer")
    assert result.score == 0.0


# ── Prompt construction ────────────────────────────────────────────────────────

def test_prompt_contains_query() -> None:
    evaluator, generate_fn = make_evaluator(valid_response())
    evaluator.evaluate("What is the refund policy?", [make_chunk()], "answer")
    prompt = generate_fn.call_args[0][0]
    assert "What is the refund policy?" in prompt


def test_prompt_contains_answer() -> None:
    evaluator, generate_fn = make_evaluator(valid_response())
    evaluator.evaluate("q", [make_chunk()], "The refund window is 30 days.")
    prompt = generate_fn.call_args[0][0]
    assert "The refund window is 30 days." in prompt


def test_prompt_contains_all_chunk_contents() -> None:
    chunks = [
        make_chunk(content="First chunk content"),
        make_chunk(content="Second chunk content"),
        make_chunk(content="Third chunk content"),
    ]
    evaluator, generate_fn = make_evaluator(valid_response())
    evaluator.evaluate("q", chunks, "answer")
    prompt = generate_fn.call_args[0][0]
    assert "First chunk content" in prompt
    assert "Second chunk content" in prompt
    assert "Third chunk content" in prompt


def test_prompt_separates_chunks_with_delimiter() -> None:
    chunks = [make_chunk(content="A"), make_chunk(content="B")]
    evaluator, generate_fn = make_evaluator(valid_response())
    evaluator.evaluate("q", chunks, "answer")
    prompt = generate_fn.call_args[0][0]
    assert "---" in prompt


def test_generate_fn_called_exactly_once() -> None:
    evaluator, generate_fn = make_evaluator(valid_response())
    evaluator.evaluate("q", [make_chunk()], "answer")
    generate_fn.assert_called_once()
