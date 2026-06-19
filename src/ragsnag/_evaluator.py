from __future__ import annotations

import json
from collections.abc import Callable

from ragsnag._models import Chunk, EvaluationResult


_PROMPT = """\
You are evaluating the quality of a RAG (retrieval-augmented generation) answer.

Question: {query}

Retrieved context:
{context}

Generated answer: {answer}

Evaluate the answer on four criteria:
1. is_grounded: Is every claim in the answer directly supported by the retrieved context? (true/false)
2. is_complete: Does the answer fully address the question without missing key parts? (true/false)
3. score: A float from 0.0 to 1.0 reflecting overall quality (grounded + complete = 1.0)
4. reason: One sentence explaining the score.

Respond with valid JSON only, no markdown, no explanation:
{{"is_grounded": true, "is_complete": true, "score": 0.95, "reason": "..."}}"""


class LLMEvaluator:
    """
    Evaluates answers using any LLM.

    Args:
        generate_fn: A callable that accepts a prompt string and returns a response string.
                     Works with any LLM — Claude, OpenAI, local models, etc.

    Example::

        client = anthropic.Anthropic()

        def generate(prompt: str) -> str:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        evaluator = LLMEvaluator(generate_fn=generate)
    """

    def __init__(self, generate_fn: Callable[[str], str]) -> None:
        self._generate = generate_fn

    def evaluate(
        self, query: str, chunks: list[Chunk], answer: str
    ) -> EvaluationResult:
        context = "\n---\n".join(c.content for c in chunks)
        prompt = _PROMPT.format(query=query, context=context, answer=answer)
        raw = self._generate(prompt)

        try:
            data = json.loads(raw.strip())
            return EvaluationResult(
                is_grounded=bool(data["is_grounded"]),
                is_complete=bool(data["is_complete"]),
                score=float(data["score"]),
                reason=str(data["reason"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return EvaluationResult(
                is_grounded=False,
                is_complete=False,
                score=0.0,
                reason=f"Evaluator returned unparseable response: {raw[:200]}",
            )
