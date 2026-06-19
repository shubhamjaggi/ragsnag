"""Shared fixtures and helpers for all tests."""
from __future__ import annotations

import pytest

from ragsnag._models import (
    Chunk,
    EvaluationResult,
    GeneratorOutput,
    LoopIteration,
    ReformulationOutput,
    ReformulationStrategy,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(
    content: str = "test content",
    source: str = "doc.pdf",
    score: float = 0.9,
) -> Chunk:
    return Chunk(content=content, source=source, score=score)


def make_evaluation(
    score: float = 0.5,
    is_grounded: bool = True,
    is_complete: bool = False,
    reason: str = "test reason",
) -> EvaluationResult:
    return EvaluationResult(
        is_grounded=is_grounded,
        is_complete=is_complete,
        score=score,
        reason=reason,
    )


def make_gen_output(answer: str = "test answer", confidence: float = 0.8) -> GeneratorOutput:
    return GeneratorOutput(answer=answer, confidence=confidence)


def make_reformulation(
    queries: list[str] | None = None,
    strategy: ReformulationStrategy = ReformulationStrategy.EXPAND,
    reasoning: str = "test reasoning",
) -> ReformulationOutput:
    return ReformulationOutput(
        queries=queries or ["reformulated query"],
        strategy=strategy,
        reasoning=reasoning,
    )


def make_iteration(
    iteration: int = 1,
    query: str = "test query",
    score: float = 0.5,
    answer: str = "test answer",
) -> LoopIteration:
    return LoopIteration(
        iteration=iteration,
        query=query,
        chunks=[make_chunk()],
        answer=answer,
        confidence=0.8,
        evaluation=make_evaluation(score=score),
    )


# ── Mock implementations ───────────────────────────────────────────────────────

class MockRetriever:
    """Returns a fixed list of chunks for any query."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or [make_chunk()]
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class SequentialRetriever:
    """Returns a different chunk list on each call."""

    def __init__(self, chunk_lists: list[list[Chunk]]) -> None:
        self.chunk_lists = chunk_lists
        self._idx = 0
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        self.calls.append((query, top_k))
        result = self.chunk_lists[min(self._idx, len(self.chunk_lists) - 1)]
        self._idx += 1
        return result[:top_k]


class MockGenerator:
    """Returns GeneratorOutputs in sequence."""

    def __init__(self, outputs: list[GeneratorOutput] | None = None) -> None:
        self.outputs = outputs or [make_gen_output()]
        self._idx = 0
        self.calls: list[tuple[str, list[Chunk]]] = []

    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
        self.calls.append((query, chunks))
        output = self.outputs[min(self._idx, len(self.outputs) - 1)]
        self._idx += 1
        return output


class MockEvaluator:
    """Returns EvaluationResults in sequence."""

    def __init__(self, results: list[EvaluationResult] | None = None) -> None:
        self.results = results or [make_evaluation()]
        self._idx = 0
        self.calls: list[tuple[str, list[Chunk], str]] = []

    def evaluate(
        self, query: str, chunks: list[Chunk], answer: str
    ) -> EvaluationResult:
        self.calls.append((query, chunks, answer))
        result = self.results[min(self._idx, len(self.results) - 1)]
        self._idx += 1
        return result


class MockReformulator:
    """Returns ReformulationOutputs in sequence."""

    def __init__(self, outputs: list[ReformulationOutput] | None = None) -> None:
        self.outputs = outputs or [make_reformulation()]
        self._idx = 0
        self.calls: list[tuple[str, list[LoopIteration]]] = []

    def reformulate(
        self, original_query: str, history: list[LoopIteration]
    ) -> ReformulationOutput:
        self.calls.append((original_query, list(history)))  # copy to prevent mutation
        output = self.outputs[min(self._idx, len(self.outputs) - 1)]
        self._idx += 1
        return output
