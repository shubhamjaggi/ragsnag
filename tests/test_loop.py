"""Tests for the core RAGLoop engine."""
from __future__ import annotations

import pytest

from ragsnag._config import LoopConfig
from ragsnag._loop import RAGLoop
from ragsnag._models import LoopIteration, ReformulationStrategy, StopReason

from tests.conftest import (
    MockEvaluator,
    MockGenerator,
    MockRetriever,
    MockReformulator,
    SequentialRetriever,
    make_chunk,
    make_evaluation,
    make_gen_output,
    make_reformulation,
)


def make_loop(
    retriever=None,
    generator=None,
    evaluator=None,
    reformulator=None,
    config=None,
) -> RAGLoop:
    return RAGLoop(
        retriever=retriever or MockRetriever(),
        generator=generator or MockGenerator(),
        evaluator=evaluator or MockEvaluator(),
        reformulator=reformulator or MockReformulator(),
        config=config or LoopConfig(),
    )


# ── Convergence ────────────────────────────────────────────────────────────────

def test_converges_on_first_iteration() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.95)])
    loop = make_loop(evaluator=evaluator, config=LoopConfig(confidence_threshold=0.8))
    result = loop.run("test query")
    assert result.stop_reason == StopReason.CONVERGED
    assert result.iterations == 1


def test_converges_on_second_iteration() -> None:
    evaluator = MockEvaluator([
        make_evaluation(score=0.4),
        make_evaluation(score=0.9),
    ])
    loop = make_loop(evaluator=evaluator, config=LoopConfig(max_iterations=3))
    result = loop.run("test query")
    assert result.stop_reason == StopReason.CONVERGED
    assert result.iterations == 2


def test_converges_at_exact_threshold() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.8)])
    loop = make_loop(evaluator=evaluator, config=LoopConfig(confidence_threshold=0.8))
    result = loop.run("test query")
    assert result.stop_reason == StopReason.CONVERGED


def test_max_iterations_when_never_converges() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.3)])
    loop = make_loop(
        evaluator=evaluator,
        config=LoopConfig(max_iterations=3, confidence_threshold=0.9),
    )
    result = loop.run("test query")
    assert result.stop_reason == StopReason.MAX_ITERATIONS
    assert result.iterations == 3


def test_max_iterations_one() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.1)])
    reformulator = MockReformulator()
    loop = make_loop(
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=1),
    )
    result = loop.run("test query")
    assert result.iterations == 1
    assert len(reformulator.calls) == 0


# ── Best result tracking ───────────────────────────────────────────────────────

def test_returns_best_answer_not_last() -> None:
    generator = MockGenerator([
        make_gen_output(answer="mediocre answer", confidence=0.6),
        make_gen_output(answer="best answer", confidence=0.9),
        make_gen_output(answer="worse answer", confidence=0.4),
    ])
    evaluator = MockEvaluator([
        make_evaluation(score=0.6),
        make_evaluation(score=0.85, is_grounded=True, is_complete=True),
        make_evaluation(score=0.3),
    ])
    loop = make_loop(
        generator=generator,
        evaluator=evaluator,
        config=LoopConfig(max_iterations=3, confidence_threshold=0.99),
    )
    result = loop.run("test query")
    assert result.answer == "best answer"
    assert result.stop_reason == StopReason.MAX_ITERATIONS


def test_best_iteration_property_on_result() -> None:
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.7),
        make_evaluation(score=0.5),
    ])
    loop = make_loop(
        evaluator=evaluator,
        config=LoopConfig(max_iterations=3, confidence_threshold=0.99),
    )
    result = loop.run("test query")
    assert result.best_iteration.evaluation.score == 0.7


# ── Trace ─────────────────────────────────────────────────────────────────────

def test_trace_contains_all_iterations() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.3)])
    loop = make_loop(
        evaluator=evaluator,
        config=LoopConfig(max_iterations=3, confidence_threshold=0.99),
    )
    result = loop.run("test query")
    assert len(result.trace) == 3


def test_trace_iterations_are_numbered_sequentially() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.3)])
    loop = make_loop(
        evaluator=evaluator,
        config=LoopConfig(max_iterations=3, confidence_threshold=0.99),
    )
    result = loop.run("test query")
    assert [it.iteration for it in result.trace] == [1, 2, 3]


def test_trace_contains_correct_answers() -> None:
    generator = MockGenerator([
        make_gen_output(answer="answer A"),
        make_gen_output(answer="answer B"),
    ])
    evaluator = MockEvaluator([make_evaluation(score=0.3)])
    loop = make_loop(
        generator=generator,
        evaluator=evaluator,
        config=LoopConfig(max_iterations=2, confidence_threshold=0.99),
    )
    result = loop.run("test query")
    assert result.trace[0].answer == "answer A"
    assert result.trace[1].answer == "answer B"


# ── Query handling ─────────────────────────────────────────────────────────────

def test_original_query_always_passed_to_generator() -> None:
    generator = MockGenerator([make_gen_output()])
    reformulator = MockReformulator([
        make_reformulation(queries=["reformulated query"])
    ])
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.9),
    ])
    loop = make_loop(
        generator=generator,
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=2),
    )
    loop.run("original query")
    for query, _ in generator.calls:
        assert query == "original query"


def test_original_query_always_passed_to_evaluator() -> None:
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.9),
    ])
    loop = make_loop(
        evaluator=evaluator,
        config=LoopConfig(max_iterations=2),
    )
    loop.run("original query")
    for query, _, _ in evaluator.calls:
        assert query == "original query"


def test_reformulated_query_used_for_retrieval() -> None:
    retriever = MockRetriever()
    reformulator = MockReformulator([
        make_reformulation(queries=["new query"])
    ])
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.9),
    ])
    loop = make_loop(
        retriever=retriever,
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=2),
    )
    loop.run("original query")
    assert retriever.calls[0][0] == "original query"
    assert retriever.calls[1][0] == "new query"


def test_reformulator_receives_original_query() -> None:
    reformulator = MockReformulator()
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.9),
    ])
    loop = make_loop(
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=2),
    )
    loop.run("original query")
    assert reformulator.calls[0][0] == "original query"


def test_reformulator_receives_full_history() -> None:
    reformulator = MockReformulator()
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.4),
        make_evaluation(score=0.9),
    ])
    loop = make_loop(
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=3),
    )
    loop.run("test query")
    assert len(reformulator.calls[0][1]) == 1
    assert len(reformulator.calls[1][1]) == 2


def test_reformulator_not_called_when_converged_first_iteration() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.95)])
    reformulator = MockReformulator()
    loop = make_loop(evaluator=evaluator, reformulator=reformulator)
    loop.run("test query")
    assert len(reformulator.calls) == 0


def test_reformulator_not_called_on_last_iteration() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.3)])
    reformulator = MockReformulator()
    loop = make_loop(
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=3, confidence_threshold=0.99),
    )
    loop.run("test query")
    assert len(reformulator.calls) == 2  # called after iter 1 and 2, not 3


# ── Multi-query (decompose) ────────────────────────────────────────────────────

def test_multi_query_chunks_are_merged() -> None:
    initial = [make_chunk(content="initial chunk", score=0.5)]
    chunks_a = [make_chunk(content="chunk from query A", score=0.9)]
    chunks_b = [make_chunk(content="chunk from query B", score=0.8)]
    # 3 entries: iter-1 single query, iter-2 "query A", iter-2 "query B"
    retriever = SequentialRetriever([initial, chunks_a, chunks_b])
    reformulator = MockReformulator([
        make_reformulation(
            queries=["query A", "query B"],
            strategy=ReformulationStrategy.DECOMPOSE,
        )
    ])
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.95),
    ])
    generator = MockGenerator([make_gen_output()])
    loop = make_loop(
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=2),
    )
    loop.run("original query")
    second_call_chunks = generator.calls[1][1]
    contents = {c.content for c in second_call_chunks}
    assert "chunk from query A" in contents
    assert "chunk from query B" in contents


def test_duplicate_chunks_are_deduplicated() -> None:
    shared_chunk = make_chunk(content="shared chunk", score=0.9)
    chunks_a = [shared_chunk, make_chunk(content="unique A", score=0.7)]
    chunks_b = [shared_chunk, make_chunk(content="unique B", score=0.6)]
    retriever = SequentialRetriever([chunks_a, chunks_b])
    reformulator = MockReformulator([
        make_reformulation(queries=["query A", "query B"])
    ])
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.95),
    ])
    generator = MockGenerator([make_gen_output()])
    loop = make_loop(
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=2),
    )
    loop.run("test query")
    second_call_chunks = generator.calls[1][1]
    contents = [c.content for c in second_call_chunks]
    assert contents.count("shared chunk") == 1


def test_merged_chunks_sorted_by_score_descending() -> None:
    chunks_a = [make_chunk(content="low score", score=0.3)]
    chunks_b = [make_chunk(content="high score", score=0.9)]
    retriever = SequentialRetriever([chunks_a, chunks_b])
    reformulator = MockReformulator([
        make_reformulation(queries=["query A", "query B"])
    ])
    evaluator = MockEvaluator([
        make_evaluation(score=0.3),
        make_evaluation(score=0.95),
    ])
    generator = MockGenerator([make_gen_output()])
    loop = make_loop(
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        reformulator=reformulator,
        config=LoopConfig(max_iterations=2),
    )
    loop.run("test query")
    second_call_chunks = generator.calls[1][1]
    scores = [c.score for c in second_call_chunks]
    assert scores == sorted(scores, reverse=True)


# ── Callback ──────────────────────────────────────────────────────────────────

def test_on_iteration_callback_called_each_iteration() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.3)])
    received: list[LoopIteration] = []
    config = LoopConfig(
        max_iterations=3,
        confidence_threshold=0.99,
        on_iteration=received.append,
    )
    loop = make_loop(evaluator=evaluator, config=config)
    loop.run("test query")
    assert len(received) == 3


def test_on_iteration_callback_receives_correct_iteration_numbers() -> None:
    evaluator = MockEvaluator([make_evaluation(score=0.3)])
    received: list[LoopIteration] = []
    config = LoopConfig(
        max_iterations=3,
        confidence_threshold=0.99,
        on_iteration=received.append,
    )
    loop = make_loop(evaluator=evaluator, config=config)
    loop.run("test query")
    assert [it.iteration for it in received] == [1, 2, 3]


def test_no_callback_does_not_raise() -> None:
    loop = make_loop(config=LoopConfig(on_iteration=None))
    result = loop.run("test query")
    assert result is not None


# ── top_k ─────────────────────────────────────────────────────────────────────

def test_top_k_passed_to_retriever() -> None:
    retriever = MockRetriever()
    loop = make_loop(retriever=retriever, config=LoopConfig(top_k=7))
    loop.run("test query")
    assert retriever.calls[0][1] == 7
