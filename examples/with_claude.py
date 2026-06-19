"""
Example: ragsnag with Claude (Anthropic) as the LLM.

Install extras:
    pip install ragsnag anthropic
"""

import anthropic

from ragsnag import (
    Chunk,
    GeneratorOutput,
    LLMEvaluator,
    LLMReformulator,
    LoopConfig,
    RAGLoop,
)


# ── Shared Claude generate function ───────────────────────────────────────────

client = anthropic.Anthropic()


def claude_generate(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text  # type: ignore[union-attr]


# ── Retriever: bring your own ──────────────────────────────────────────────────
# ragsnag does not manage your vector database.
# Implement the Retriever protocol against whichever store you use.

class MyRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        # Replace with your actual vector DB call:
        # results = pinecone_index.query(vector=embed(query), top_k=top_k)
        # return [Chunk(content=r.metadata["text"], source=r.id, score=r.score) for r in results.matches]
        return [
            Chunk(content="Placeholder document chunk.", source="docs/policy.pdf", score=0.75)
        ]


# ── Generator: Claude answers the question ────────────────────────────────────

class ClaudeGenerator:
    _PROMPT = (
        "Answer the following question using ONLY the context provided below. "
        "If the context does not contain enough information, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer concisely. End your response with: CONFIDENCE: <0.0-1.0>"
    )

    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
        context = "\n---\n".join(c.content for c in chunks)
        raw = claude_generate(self._PROMPT.format(context=context, query=query))

        confidence = 0.7
        answer = raw
        if "CONFIDENCE:" in raw:
            parts = raw.rsplit("CONFIDENCE:", 1)
            answer = parts[0].strip()
            try:
                confidence = float(parts[1].strip())
            except ValueError:
                pass

        return GeneratorOutput(answer=answer, confidence=min(max(confidence, 0.0), 1.0))


# ── Wire it up ────────────────────────────────────────────────────────────────

loop = RAGLoop(
    retriever=MyRetriever(),
    generator=ClaudeGenerator(),
    evaluator=LLMEvaluator(generate_fn=claude_generate),
    reformulator=LLMReformulator(generate_fn=claude_generate),
    config=LoopConfig(
        max_iterations=3,
        confidence_threshold=0.85,
        top_k=5,
        on_iteration=lambda it: print(
            f"  Iter {it.iteration}: score={it.evaluation.score:.2f} | {it.evaluation.reason}"
        ),
    ),
)

if __name__ == "__main__":
    result = loop.run("What is the refund policy for international orders?")

    print(f"\nAnswer: {result.answer}")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Converged in {result.iterations} iteration(s)")
