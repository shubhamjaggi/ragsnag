from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragsnag._models import (
    Chunk,
    EvaluationResult,
    GeneratorOutput,
    LoopIteration,
    ReformulationOutput,
)


@runtime_checkable
class Retriever(Protocol):
    """Fetches relevant chunks from a document store given a query string.

    ragsnag calls retrieve() once per iteration (or once per sub-query when
    the DECOMPOSE strategy is active). You implement this against whichever
    vector store, BM25 index, or hybrid search system you use.

    For ``isinstance(obj, Retriever)`` to pass, the object only needs a
    ``retrieve`` method — Python's runtime protocol check does not verify
    argument types, return type, or default values. Full signature conformance
    (including the ``top_k`` default of 5) is enforced by static type checkers
    (mypy, pyright), not by isinstance.

    Implementation notes:
    - Convert your vector store's result objects into ``Chunk`` instances.
      Map the text to ``content``, the document identifier to ``source``,
      and the similarity score to ``score`` (must be in [0.0, 1.0]).
    - If your vector store returns scores outside [0, 1] (e.g. raw dot
      products), normalize them before constructing Chunk.
    - The ``top_k`` parameter comes from ``LoopConfig.top_k``. You do not
      need to enforce it yourself if your underlying client already limits
      results, but it is passed for convenience.

    Example::

        class PineconeRetriever:
            def __init__(self, index: pinecone.Index, embed_fn) -> None:
                self._index = index
                self._embed = embed_fn

            def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
                vector = self._embed(query)
                results = self._index.query(
                    vector=vector, top_k=top_k, include_metadata=True
                )
                return [
                    Chunk(
                        content=match.metadata["text"],
                        source=match.id,
                        score=match.score,
                    )
                    for match in results.matches
                ]
    """

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]: ...


@runtime_checkable
class Generator(Protocol):
    """Produces an answer given a query and a list of retrieved chunks.

    ragsnag always passes the ORIGINAL user query to generate(), not the
    reformulated query. This is intentional: the reformulated query is only
    used to find better chunks. The actual question being answered never
    changes. Passing a reformulated query to the Generator would cause it to
    answer a subtly different question than the user asked.

    Implementation notes:
    - Build a prompt that includes all chunk contents as context and the
      original query as the question.
    - Return a ``confidence`` value between 0.0 and 1.0. You can ask the
      LLM to self-report confidence ("End with CONFIDENCE: 0.85"), derive
      it from token log-probabilities, or use a fixed value like 0.7.
      ragsnag stores this in the trace but does not use it for loop control —
      only the Evaluator's score controls whether to keep looping.
    - Keep the prompt focused: instruct the model to answer using only the
      provided context, so that hallucination is minimised.

    Example::

        class ClaudeGenerator:
            _PROMPT = (
                "Answer using ONLY the context below.\\n\\n"
                "Context:\\n{context}\\n\\n"
                "Question: {query}\\n\\n"
                "Answer concisely. End with: CONFIDENCE: <0.0-1.0>"
            )

            def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
                context = "\\n---\\n".join(c.content for c in chunks)
                raw = call_claude(self._PROMPT.format(context=context, query=query))
                answer, confidence = parse_confidence(raw)
                return GeneratorOutput(answer=answer, confidence=confidence)
    """

    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores whether an answer is grounded in the retrieved chunks and complete.

    This is the quality gate of the loop. ragsnag compares EvaluationResult.score
    against LoopConfig.confidence_threshold after every iteration. If the score
    meets or exceeds the threshold, the loop stops. If not, the Reformulator
    is called to improve the query.

    ragsnag always passes the ORIGINAL user query to evaluate(), not the
    reformulated query. The evaluation should always answer: "Does this answer
    correctly address what the user originally asked?"

    The ``reason`` field in EvaluationResult is particularly important — it is
    the primary input to the Reformulator. A precise reason ("chunks describe
    domestic policy only; question asks about international orders") leads to
    better strategy selection than a vague one ("answer is incomplete").

    Implementation notes:
    - Score on a consistent scale. If your threshold is 0.8, your scoring
      logic should produce 0.8 or higher only for answers you'd actually
      ship to a user.
    - is_grounded and is_complete are independent. An answer can be grounded
      but incomplete (answers part of the question accurately), or complete
      but not grounded (answers fully but from the model's parametric
      knowledge, not from the chunks).
    - If you implement a heuristic evaluator (no LLM), be careful: simple
      keyword overlap can give false high scores for irrelevant content that
      happens to share vocabulary with the query.

    Example::

        class MyEvaluator:
            def evaluate(
                self, query: str, chunks: list[Chunk], answer: str
            ) -> EvaluationResult:
                # Ask an LLM to judge the answer
                verdict = call_llm(build_eval_prompt(query, chunks, answer))
                return EvaluationResult(
                    is_grounded=verdict["grounded"],
                    is_complete=verdict["complete"],
                    score=verdict["score"],
                    reason=verdict["reason"],
                )
    """

    def evaluate(
        self, query: str, chunks: list[Chunk], answer: str
    ) -> EvaluationResult: ...


@runtime_checkable
class Reformulator(Protocol):
    """Rewrites the query when the previous iteration's answer was not good enough.

    The Reformulator is called after every iteration where the Evaluator's
    score is below LoopConfig.confidence_threshold, EXCEPT on the final
    iteration (reformulating after the last attempt wastes an LLM call since
    there are no more iterations to use the result).

    It receives the ORIGINAL user query and the full history of all iterations
    so far. It returns one or more new queries to use in the next iteration.

    The key input for strategy selection is the ``reason`` string from the
    most recent EvaluationResult in history. A well-written reason string
    (e.g. "vocabulary mismatch", "multi-part question", "too generic") enables
    the Reformulator to pick the right strategy.

    When the Reformulator returns multiple queries (DECOMPOSE strategy), ragsnag
    calls retrieve() for each, merges the chunks, deduplicates by content,
    sorts by score descending, and passes the combined set to the Generator.

    Implementation notes:
    - Always return at least one query (ReformulationOutput.queries has a
      minimum length of 1).
    - The returned query replaces the RETRIEVAL query for the next iteration.
      The Generator and Evaluator still receive the original user query.
    - Use ``history[-1].evaluation.reason`` as the primary signal for what
      went wrong and what to fix.
    - The full history lets you avoid repeating strategies that already failed.

    Example::

        from ragsnag import ReformulationStrategy

        class MyReformulator:
            def reformulate(
                self, original_query: str, history: list[LoopIteration]
            ) -> ReformulationOutput:
                last_reason = history[-1].evaluation.reason
                new_query = call_llm(
                    build_reformulation_prompt(original_query, history)
                )
                return ReformulationOutput(
                    queries=[new_query],
                    strategy=ReformulationStrategy.EXPAND,
                    reasoning="Added synonyms based on vocabulary mismatch.",
                )
    """

    def reformulate(
        self, original_query: str, history: list[LoopIteration]
    ) -> ReformulationOutput: ...
