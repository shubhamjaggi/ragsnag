from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single piece of text retrieved from a document store.

    Chunks are the unit of retrieval. Your Retriever breaks documents into
    chunks and returns the most relevant ones for a given query. ragsnag
    passes those chunks to the Generator as context.

    Attributes:
        content:  The raw text of the chunk.
        source:   Where the chunk came from — a file path, URL, document ID,
                  or any string that helps you trace it back to the original.
        score:    How closely this chunk matched the search query, as reported
                  by your vector store (0.0 = no match, 1.0 = perfect match).
                  ragsnag uses this to sort merged chunks in multi-query mode.
        metadata: Any extra data you want to carry — page number, section
                  heading, timestamp, etc. Not used by ragsnag internally.
    """

    content: str
    source: str
    score: float = Field(ge=0.0, le=1.0, description="Retrieval similarity score")
    metadata: dict[str, object] = Field(default_factory=dict)


class GeneratorOutput(BaseModel):
    """What your Generator returns for a single generate call.

    Attributes:
        answer:     The generated answer text.
        confidence: How confident the generator is in the answer (0.0–1.0).
                    This is self-reported by the model, not computed by ragsnag.
                    You can ask the LLM to rate its own confidence, derive it
                    from log-probabilities, or use a fixed default. ragsnag
                    stores this in the trace but does NOT use it to decide
                    whether to keep looping — only the Evaluator's score does.
    """

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    """The evaluator's verdict on a single generate attempt.

    The score is the only value ragsnag uses to decide whether to stop or
    keep looping. is_grounded and is_complete are diagnostic — they tell
    you *why* the score is what it is.

    Attributes:
        is_grounded: True if every claim in the answer is directly supported
                     by the retrieved chunks. False if the answer contains
                     information that wasn't in the chunks (hallucination).
        is_complete: True if the answer fully addresses all parts of the
                     question. False if the answer is partial or only covers
                     part of what was asked.
        score:       Overall quality score from 0.0 to 1.0. ragsnag compares
                     this against LoopConfig.confidence_threshold to decide
                     whether to stop. A typical scoring guide:
                       1.0 — grounded and complete
                       0.7 — grounded but incomplete, or complete but partially
                              hallucinated
                       0.3 — partially grounded, missing key information
                       0.0 — not grounded, wrong answer, or evaluator failed
        reason:      One sentence explaining the score. The Reformulator reads
                     this to decide which reformulation strategy to apply. A
                     clear, specific reason leads to better query reformulation.
                     Example: "Chunks describe domestic policy only; question
                     asks about international orders specifically."
    """

    is_grounded: bool = Field(description="Answer is supported by the retrieved chunks")
    is_complete: bool = Field(description="Answer fully addresses the question")
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Why this score was assigned")


class ReformulationStrategy(str, Enum):
    """The approach the Reformulator uses to rewrite the query.

    Each strategy targets a different root cause of retrieval failure:

    EXPAND:
        Add synonyms, related terms, and broader vocabulary. Use when
        retrieval returns some relevant content but misses related concepts.
        Example: "refund policy" → "refund return reimbursement money-back policy"

    NARROW:
        Make the query more specific. Use when retrieved chunks are too broad
        or cover many topics at a surface level without depth.
        Example: "employee benefits" → "parental leave weeks paid policy"

    DECOMPOSE:
        Split a multi-part question into separate sub-queries, run each
        independently, then merge all retrieved chunks before generating.
        Use when one query can't cover two distinct topics at once.
        Example: "pricing and billing cycle"
              → ["pricing tiers", "billing annual monthly"]

    STEP_BACK:
        Ask a broader, more general version of the question first. Use when
        the specific query finds nothing — retrieving broader context may
        surface the right document section.
        Example: "maximum upload size for free tier mobile users"
              → "file upload size limits"

    HYDE (Hypothetical Document Embedding):
        Generate a short hypothetical answer that looks like what the source
        document might say, then search for that instead of the question.
        Use when the question is abstract or conceptual and doesn't match
        how the documents are written (documents are written as answers,
        not questions).
        Example query: "What causes API latency spikes?"
        HyDE searches: "API latency spikes are caused by connection pool
        exhaustion, cold starts, or N+1 query patterns."

    PERSPECTIVE_SHIFT:
        Reframe the query using completely different vocabulary. Use when
        the user's words don't match the terminology in the source documents.
        Example: "how to cancel" → "account termination close subscription"
    """

    EXPAND = "expand"
    NARROW = "narrow"
    DECOMPOSE = "decompose"
    STEP_BACK = "step_back"
    HYDE = "hyde"
    PERSPECTIVE_SHIFT = "perspective_shift"


class ReformulationOutput(BaseModel):
    """What a Reformulator returns to guide the next retrieval.

    Attributes:
        queries:   One or more query strings to use in the next iteration.
                   Most strategies return a single rewritten query. DECOMPOSE
                   returns multiple — ragsnag runs retrieve() for each, merges
                   all results, deduplicates by content, and sorts by score
                   before passing to the Generator.
        strategy:  Which strategy was applied. Stored in the trace for
                   debugging — lets you see which strategies fired and when.
        reasoning: Why this strategy was chosen. One sentence. Useful for
                   understanding and auditing the reformulation logic.
    """

    queries: list[str] = Field(min_length=1)
    strategy: ReformulationStrategy
    reasoning: str


class LoopIteration(BaseModel):
    """A complete record of one attempt within the loop.

    Every call to RAGLoop.run() produces one LoopIteration per loop cycle.
    All iterations are collected into LoopResult.trace, giving you full
    visibility into what happened at each step.

    Attributes:
        iteration:  The iteration number, starting at 1.
        query:      The query string(s) used for retrieval in this iteration.
                    On iteration 1 this is the original user query. On
                    subsequent iterations this is the reformulated query.
                    If DECOMPOSE produced multiple queries, they are joined
                    with " | " for display.
        chunks:     The chunks that were retrieved and passed to the Generator.
                    In multi-query mode, this is the merged, deduplicated,
                    score-sorted set of chunks from all sub-queries.
        answer:     The answer the Generator produced from this iteration's
                    chunks.
        confidence: The Generator's self-reported confidence in its answer.
        evaluation: The Evaluator's verdict — score, grounding, completeness,
                    and the reason string that guides the next reformulation.
    """

    iteration: int = Field(ge=1)
    query: str
    chunks: list[Chunk]
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    evaluation: EvaluationResult


class StopReason(str, Enum):
    """Why the loop stopped.

    CONVERGED:
        The Evaluator's score reached or exceeded LoopConfig.confidence_threshold.
        This is the success case — the loop found an answer it's confident in.

    MAX_ITERATIONS:
        The loop ran LoopConfig.max_iterations times without converging.
        ragsnag returns the best answer found across all iterations (not
        necessarily the last one). Check result.best_iteration to see which
        iteration produced the returned answer.

    HUMAN_APPROVED:
        Reserved for future human-in-the-loop workflows.

    ERROR:
        An unrecoverable error occurred. Currently unused — exceptions
        propagate directly. Reserved for future error-handling extensions.
    """

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    HUMAN_APPROVED = "human_approved"
    ERROR = "error"


class LoopResult(BaseModel):
    """The final output of RAGLoop.run().

    Attributes:
        answer:      The best answer found across all iterations. If the loop
                     converged, this is the answer from the converging iteration.
                     If max_iterations was hit, this is the answer from whichever
                     iteration had the highest evaluation score — not the last
                     iteration's answer.
        confidence:  The Generator's self-reported confidence for the returned
                     answer. Comes from GeneratorOutput.confidence, not from
                     the Evaluator. Use evaluation.score on best_iteration for
                     the Evaluator's quality score.
        iterations:  How many loop iterations ran before stopping.
        stop_reason: Why the loop stopped. CONVERGED means success.
                     MAX_ITERATIONS means the threshold was never reached —
                     inspect the trace to understand why.
        trace:       Every LoopIteration that ran, in order. Use this to debug
                     retrieval failures, see which queries were tried, what
                     chunks were returned, and why each evaluation failed.

    Properties:
        best_iteration: The LoopIteration with the highest evaluation score.
                        When stop_reason is CONVERGED, this is also the last
                        iteration. When stop_reason is MAX_ITERATIONS, this
                        may be an earlier iteration — the answer field on
                        LoopResult reflects this iteration's answer.
    """

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    iterations: int = Field(ge=1)
    stop_reason: StopReason
    trace: list[LoopIteration]

    @property
    def best_iteration(self) -> LoopIteration:
        return max(self.trace, key=lambda it: it.evaluation.score)
