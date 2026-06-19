from __future__ import annotations

from ragsnag._config import LoopConfig
from ragsnag._models import Chunk, LoopIteration, LoopResult, StopReason
from ragsnag._protocols import Evaluator, Generator, Reformulator, Retriever


class RAGLoop:
    """The core loop engine. Orchestrates iterative retrieval-augmented generation.

    RAGLoop runs a retrieve → generate → evaluate → reformulate cycle until
    one of two conditions is met:
    - The Evaluator's score reaches LoopConfig.confidence_threshold (CONVERGED)
    - The loop has run LoopConfig.max_iterations times (MAX_ITERATIONS)

    ragsnag owns only the loop logic. It has no opinion on which LLM, vector
    store, or retrieval strategy you use. Every component is pluggable via
    the four protocols: Retriever, Generator, Evaluator, Reformulator.

    Args:
        retriever:    Fetches chunks from your document store. Called once per
                      iteration (or once per sub-query for DECOMPOSE strategy).
        generator:    Produces an answer from the original query + retrieved chunks.
                      Always receives the original user query, never the
                      reformulated one — reformulation only affects retrieval.
        evaluator:    Scores the answer's grounding and completeness. Returns a
                      score that controls whether the loop continues. Also
                      returns a reason string that guides the next reformulation.
        reformulator: Rewrites the retrieval query when the evaluation score is
                      below the threshold. Has access to the full iteration
                      history including all prior evaluation reasons.
        config:       Controls max iterations, confidence threshold, top-k chunks,
                      and an optional per-iteration callback. Defaults to
                      LoopConfig() if not provided.

    Example::

        loop = RAGLoop(
            retriever=MyRetriever(),
            generator=MyGenerator(),
            evaluator=LLMEvaluator(generate_fn=my_llm_fn),
            reformulator=LLMReformulator(generate_fn=my_llm_fn),
            config=LoopConfig(max_iterations=3, confidence_threshold=0.85),
        )
        result = loop.run("What is the late payment fee on overdue invoices?")
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        evaluator: Evaluator,
        reformulator: Reformulator,
        config: LoopConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator
        self.reformulator = reformulator
        self.config = config or LoopConfig()

    def run(self, query: str) -> LoopResult:
        """Run the retrieval loop for the given query.

        Executes up to LoopConfig.max_iterations of:
          1. retrieve: fetch chunks using the current query (or sub-queries)
          2. generate: produce an answer from the original query + chunks
          3. evaluate: score the answer's quality
          4. reformulate: rewrite the query if the score is below threshold
                          (skipped on the last iteration)

        The original ``query`` is always passed to generate() and evaluate().
        Only the retrieval step uses the reformulated query. This ensures the
        Generator and Evaluator always work against the user's actual intent,
        regardless of how the retrieval query was transformed.

        The best answer across all iterations is returned — not necessarily
        the last one. If iteration 2 scores 0.9 and iteration 3 scores 0.7,
        the answer from iteration 2 is returned.

        Args:
            query: The user's question, unchanged throughout all iterations.

        Returns:
            LoopResult with the best answer found, the stop reason, and the
            full iteration trace for debugging.
        """
        history: list[LoopIteration] = []
        current_queries = [query]
        best: LoopIteration | None = None

        for i in range(self.config.max_iterations):
            chunks = self._retrieve_and_merge(current_queries)
            gen = self.generator.generate(query, chunks)
            evaluation = self.evaluator.evaluate(query, chunks, gen.answer)

            iteration = LoopIteration(
                iteration=i + 1,
                query=" | ".join(current_queries),
                chunks=chunks,
                answer=gen.answer,
                confidence=gen.confidence,
                evaluation=evaluation,
            )
            history.append(iteration)

            if self.config.on_iteration:
                self.config.on_iteration(iteration)

            if best is None or evaluation.score > best.evaluation.score:
                best = iteration

            if evaluation.score >= self.config.confidence_threshold:
                return LoopResult(
                    answer=gen.answer,
                    confidence=gen.confidence,
                    iterations=i + 1,
                    stop_reason=StopReason.CONVERGED,
                    trace=history,
                )

            # Do not reformulate after the last iteration — there are no more
            # iterations to use the result, so it would be a wasted LLM call.
            if i < self.config.max_iterations - 1:
                reformulation = self.reformulator.reformulate(query, history)
                current_queries = reformulation.queries

        assert best is not None
        return LoopResult(
            answer=best.answer,
            confidence=best.confidence,
            iterations=len(history),
            stop_reason=StopReason.MAX_ITERATIONS,
            trace=history,
        )

    def _retrieve_and_merge(self, queries: list[str]) -> list[Chunk]:
        """Retrieve chunks for all queries, deduplicate by content, sort by score.

        When the DECOMPOSE strategy returns multiple sub-queries, this method
        runs retrieve() for each one independently and merges the results.
        Deduplication is by exact content string match — the same chunk
        appearing in multiple query results is included only once (first
        occurrence wins, since results are already sorted by relevance within
        each query). The merged list is sorted by score descending so the
        Generator receives the most relevant chunks first.
        """
        seen: set[str] = set()
        chunks: list[Chunk] = []
        for q in queries:
            for chunk in self.retriever.retrieve(q, top_k=self.config.top_k):
                if chunk.content not in seen:
                    seen.add(chunk.content)
                    chunks.append(chunk)
        return sorted(chunks, key=lambda c: c.score, reverse=True)
