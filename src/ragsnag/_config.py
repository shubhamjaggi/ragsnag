from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ragsnag._models import LoopIteration


@dataclass
class LoopConfig:
    """Controls how the RAGLoop behaves across iterations.

    All fields have sensible defaults for general use. Start with the defaults
    and tune based on what you observe in LoopResult.trace.

    Args:
        max_iterations:
            Maximum number of retrieve → generate → evaluate cycles to run
            before giving up. Default: 3.

            Rationale: Each iteration costs one retrieval call + one LLM
            generate call + one LLM evaluate call. Beyond 3 iterations the
            marginal improvement per iteration typically decreases while cost
            grows linearly. If your evaluator scores are still rising at
            iteration 3, increase this. If the loop almost always converges
            at iteration 1, you can lower it.

            The loop stops early if confidence_threshold is reached, so
            setting max_iterations higher does not force more LLM calls —
            it only raises the ceiling.

        confidence_threshold:
            The minimum Evaluator score (0.0–1.0) that counts as a good
            enough answer. When a score meets or exceeds this threshold the
            loop stops immediately and returns CONVERGED. Default: 0.8.

            Rationale: 0.8 is a deliberately conservative default. It means
            the Evaluator must rate the answer as both mostly grounded and
            mostly complete before stopping. Lower this (e.g. 0.65) if your
            use case tolerates partial answers and you want fewer iterations.
            Raise it (e.g. 0.95) for high-stakes Q&A where hallucinations are
            unacceptable and you'd rather loop more than return a mediocre
            answer.

            This threshold is compared against EvaluationResult.score, which
            your Evaluator implementation controls. Make sure your scoring
            scale is consistent with this threshold.

        top_k:
            How many chunks to retrieve per query per iteration. Default: 5.

            Rationale: Passing more chunks gives the Generator more context
            but also increases prompt size (cost + latency) and introduces
            more noise. 5 is a practical default for most document stores.
            Increase to 10+ for large, dense corpora where the answer is
            likely spread across many chunks. Lower to 3 for fast, cheap
            retrieval when documents are short and focused.

            In multi-query mode (DECOMPOSE strategy), top_k is applied per
            sub-query. If DECOMPOSE returns 2 sub-queries, the Generator
            receives up to 2 × top_k chunks (after deduplication).

        on_iteration:
            An optional callback function called after every iteration,
            immediately after the Evaluator scores the answer. Signature:
            ``fn(iteration: LoopIteration) -> None``. Default: None.

            Use this for:
            - Logging / observability: print scores as they happen
            - Progress indicators: update a UI spinner
            - Early exit signals: raise an exception to abort mid-loop
            - Streaming partial results to a caller

            The callback receives the full LoopIteration object, including
            the query used, chunks retrieved, answer generated, and the
            evaluation result with score and reason.

    Example::

        config = LoopConfig(
            max_iterations=5,
            confidence_threshold=0.9,
            top_k=8,
            on_iteration=lambda it: print(
                f"[iter {it.iteration}] score={it.evaluation.score:.2f} "
                f"| {it.evaluation.reason}"
            ),
        )
    """

    max_iterations: int = 3
    confidence_threshold: float = 0.8
    top_k: int = 5
    on_iteration: Callable[[LoopIteration], None] | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
