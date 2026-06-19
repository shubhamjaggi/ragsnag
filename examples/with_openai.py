"""
Example: ragsnag with OpenAI as the LLM.

Install extras:
    pip install ragsnag openai
"""

import openai

from ragsnag import (
    Chunk,
    GeneratorOutput,
    HeuristicReformulator,
    LLMEvaluator,
    LLMReformulator,
    LoopConfig,
    RAGLoop,
)


# ── Shared OpenAI generate function ───────────────────────────────────────────

oai = openai.OpenAI()


def openai_generate(prompt: str) -> str:
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


# ── Retriever ──────────────────────────────────────────────────────────────────

class MyRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        # Replace with your actual vector DB call
        return [
            Chunk(content="Placeholder document chunk.", source="docs/policy.pdf", score=0.75)
        ]


# ── Generator ──────────────────────────────────────────────────────────────────

class OpenAIGenerator:
    _PROMPT = (
        "Answer using ONLY the context below. "
        "If insufficient, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer concisely. End with: CONFIDENCE: <0.0-1.0>"
    )

    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
        context = "\n---\n".join(c.content for c in chunks)
        raw = openai_generate(self._PROMPT.format(context=context, query=query))

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
    generator=OpenAIGenerator(),
    evaluator=LLMEvaluator(generate_fn=openai_generate),
    reformulator=LLMReformulator(generate_fn=openai_generate),
    config=LoopConfig(max_iterations=3, confidence_threshold=0.85),
)

if __name__ == "__main__":
    result = loop.run("What is the refund policy for international orders?")
    print(f"Answer: {result.answer}")
    print(f"Stop reason: {result.stop_reason} | Iterations: {result.iterations}")
