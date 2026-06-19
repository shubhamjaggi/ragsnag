"""
Example: ragsnag with fully custom Evaluator and Reformulator.

Use this when you want complete control over evaluation and reformulation
logic without relying on an LLM for those steps.
"""

from ragsnag import (
    Chunk,
    EvaluationResult,
    GeneratorOutput,
    LoopConfig,
    LoopIteration,
    RAGLoop,
    ReformulationOutput,
    ReformulationStrategy,
)


# ── Custom Evaluator: keyword overlap heuristic ───────────────────────────────

class KeywordEvaluator:
    """
    Scores answers by measuring keyword overlap with retrieved chunks.
    Free, fast, zero LLM calls — good for high-volume or cost-sensitive use.
    """

    def evaluate(
        self, query: str, chunks: list[Chunk], answer: str
    ) -> EvaluationResult:
        if not chunks or not answer.strip():
            return EvaluationResult(
                is_grounded=False,
                is_complete=False,
                score=0.0,
                reason="No chunks retrieved or empty answer.",
            )

        all_chunk_text = " ".join(c.content.lower() for c in chunks)
        answer_words = set(answer.lower().split())
        chunk_words = set(all_chunk_text.split())
        stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "for"}
        answer_keywords = answer_words - stopwords

        if not answer_keywords:
            return EvaluationResult(
                is_grounded=False, is_complete=False, score=0.0,
                reason="Answer contains only stopwords.",
            )

        overlap = answer_keywords & chunk_words
        grounding_score = len(overlap) / len(answer_keywords)
        is_grounded = grounding_score >= 0.5

        query_words = set(query.lower().split()) - stopwords
        completeness_score = len(query_words & answer_words) / max(len(query_words), 1)
        is_complete = completeness_score >= 0.4

        score = round((grounding_score * 0.7) + (completeness_score * 0.3), 3)

        return EvaluationResult(
            is_grounded=is_grounded,
            is_complete=is_complete,
            score=min(score, 1.0),
            reason=(
                f"Keyword overlap: {len(overlap)}/{len(answer_keywords)} answer words "
                f"found in chunks (grounding={grounding_score:.2f}, completeness={completeness_score:.2f})."
            ),
        )


# ── Custom Reformulator: rule-based ───────────────────────────────────────────

class RuleBasedReformulator:
    """
    Expands the query by appending domain-specific synonyms.
    No LLM required.
    """

    _SYNONYMS: dict[str, list[str]] = {
        "refund": ["return", "reimbursement", "money back"],
        "cancel": ["terminate", "close", "end subscription"],
        "price": ["cost", "fee", "pricing", "rate"],
        "policy": ["terms", "rules", "guidelines"],
        "international": ["global", "overseas", "cross-border", "foreign"],
    }

    def reformulate(
        self, original_query: str, history: list[LoopIteration]
    ) -> ReformulationOutput:
        extra_terms: list[str] = []
        for word in original_query.lower().split():
            if word in self._SYNONYMS:
                extra_terms.extend(self._SYNONYMS[word])

        if extra_terms:
            new_query = f"{original_query} {' '.join(extra_terms)}"
            return ReformulationOutput(
                queries=[new_query],
                strategy=ReformulationStrategy.EXPAND,
                reasoning=f"Appended synonyms: {extra_terms}",
            )

        return ReformulationOutput(
            queries=[original_query],
            strategy=ReformulationStrategy.EXPAND,
            reasoning="No known synonyms found; keeping original query.",
        )


# ── Minimal generator stub ────────────────────────────────────────────────────

class EchoGenerator:
    """Stub — replace with your real LLM generator."""

    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
        context_snippet = chunks[0].content[:100] if chunks else "no context"
        return GeneratorOutput(
            answer=f"Based on the documents: {context_snippet}",
            confidence=0.6,
        )


class EchoRetriever:
    """Stub — replace with your real vector DB retriever."""

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        return [Chunk(content=f"Document relevant to: {query}", source="stub", score=0.7)]


# ── Wire it up ────────────────────────────────────────────────────────────────

loop = RAGLoop(
    retriever=EchoRetriever(),
    generator=EchoGenerator(),
    evaluator=KeywordEvaluator(),
    reformulator=RuleBasedReformulator(),
    config=LoopConfig(max_iterations=2, confidence_threshold=0.75),
)

if __name__ == "__main__":
    result = loop.run("What is the refund policy for international orders?")
    print(f"Answer: {result.answer}")
    print(f"Stop reason: {result.stop_reason} | Iterations: {result.iterations}")
    print("\nTrace:")
    for it in result.trace:
        print(f"  [{it.iteration}] score={it.evaluation.score} | {it.evaluation.reason}")
