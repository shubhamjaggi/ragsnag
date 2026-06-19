"""Tests for Protocol conformance."""
from __future__ import annotations

from ragsnag._models import (
    Chunk,
    EvaluationResult,
    GeneratorOutput,
    LoopIteration,
    ReformulationOutput,
    ReformulationStrategy,
)
from ragsnag._protocols import Evaluator, Generator, Reformulator, Retriever

from tests.conftest import make_chunk, make_evaluation, make_reformulation


# ── Conforming implementations ────────────────────────────────────────────────

class GoodRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        return []


class GoodGenerator:
    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
        return GeneratorOutput(answer="", confidence=1.0)


class GoodEvaluator:
    def evaluate(self, query: str, chunks: list[Chunk], answer: str) -> EvaluationResult:
        return make_evaluation()


class GoodReformulator:
    def reformulate(
        self, original_query: str, history: list[LoopIteration]
    ) -> ReformulationOutput:
        return make_reformulation()


def test_retriever_protocol_satisfied() -> None:
    assert isinstance(GoodRetriever(), Retriever)


def test_generator_protocol_satisfied() -> None:
    assert isinstance(GoodGenerator(), Generator)


def test_evaluator_protocol_satisfied() -> None:
    assert isinstance(GoodEvaluator(), Evaluator)


def test_reformulator_protocol_satisfied() -> None:
    assert isinstance(GoodReformulator(), Reformulator)


# ── Non-conforming implementations ────────────────────────────────────────────

class NoMethodRetriever:
    pass


class NoMethodGenerator:
    pass


class NoMethodEvaluator:
    pass


class NoMethodReformulator:
    pass


def test_retriever_protocol_not_satisfied_without_method() -> None:
    assert not isinstance(NoMethodRetriever(), Retriever)


def test_generator_protocol_not_satisfied_without_method() -> None:
    assert not isinstance(NoMethodGenerator(), Generator)


def test_evaluator_protocol_not_satisfied_without_method() -> None:
    assert not isinstance(NoMethodEvaluator(), Evaluator)


def test_reformulator_protocol_not_satisfied_without_method() -> None:
    assert not isinstance(NoMethodReformulator(), Reformulator)


# ── Public __init__ exports ───────────────────────────────────────────────────

def test_public_api_exports_protocols() -> None:
    import ragsnag
    assert hasattr(ragsnag, "Retriever")
    assert hasattr(ragsnag, "Generator")
    assert hasattr(ragsnag, "Evaluator")
    assert hasattr(ragsnag, "Reformulator")


def test_public_api_exports_core_classes() -> None:
    import ragsnag
    assert hasattr(ragsnag, "RAGLoop")
    assert hasattr(ragsnag, "LoopConfig")
    assert hasattr(ragsnag, "LLMEvaluator")
    assert hasattr(ragsnag, "LLMReformulator")
    assert hasattr(ragsnag, "HeuristicReformulator")


def test_public_api_exports_models() -> None:
    import ragsnag
    for name in ["Chunk", "GeneratorOutput", "EvaluationResult",
                 "LoopIteration", "LoopResult", "ReformulationOutput",
                 "ReformulationStrategy", "StopReason"]:
        assert hasattr(ragsnag, name), f"Missing export: {name}"
