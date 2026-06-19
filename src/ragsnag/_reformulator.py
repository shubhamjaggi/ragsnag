from __future__ import annotations

import json
from collections.abc import Callable

from ragsnag._models import (
    LoopIteration,
    ReformulationOutput,
    ReformulationStrategy,
)

_STRATEGY_HINTS: dict[ReformulationStrategy, list[str]] = {
    ReformulationStrategy.PERSPECTIVE_SHIFT: [
        "vocabulary",
        "terminology",
        "different words",
        "jargon",
        "synonym",
        "no chunk",
        "nothing found",
        "not retrieved",
    ],
    ReformulationStrategy.NARROW: [
        "too generic",
        "too broad",
        "surface level",
        "vague",
        "general",
        "incomplete",
        "partial",
        "missing specific",
    ],
    ReformulationStrategy.DECOMPOSE: [
        "multi-part",
        "multiple question",
        "two question",
        "missing second",
        "missing third",
        "also ask",
        "additionally",
    ],
    ReformulationStrategy.STEP_BACK: [
        "no relevant",
        "completely off",
        "unrelated",
        "abstract",
    ],
    ReformulationStrategy.HYDE: [
        "conceptual",
        "how does",
        "explain",
        "mechanism",
        "describe",
    ],
}

_PROMPT = """\
You are improving a search query to retrieve better information for a RAG system.

Original question: {query}

What was tried and why it failed:
{history_summary}

Select a reformulation strategy:
- expand: add synonyms and related terms to cast a wider net
- narrow: be more specific to get more targeted results
- decompose: split into sub-queries (use when question has multiple parts)
- step_back: ask a broader version of the question first
- hyde: write a short hypothetical answer that looks like what the source
  document might say, then search for that
- perspective_shift: reframe using completely different vocabulary

Respond with valid JSON only:
{{
  "strategy": "<one of: expand, narrow, decompose, step_back, hyde, perspective_shift>",
  "queries": ["<query 1>", "<optional query 2 if decompose>"],
  "reasoning": "<one sentence explaining your choice>"
}}"""


def _select_strategy_heuristic(reason: str) -> ReformulationStrategy:
    reason_lower = reason.lower()
    for strategy, hints in _STRATEGY_HINTS.items():
        if any(h in reason_lower for h in hints):
            return strategy
    return ReformulationStrategy.EXPAND


def _build_history_summary(history: list[LoopIteration]) -> str:
    lines = []
    for it in history:
        lines.append(
            f"Iteration {it.iteration}:\n"
            f"  Query used: {it.query}\n"
            f"  Answer: {it.answer[:300]}\n"
            f"  Score: {it.evaluation.score}\n"
            f"  Why it failed: {it.evaluation.reason}"
        )
    return "\n\n".join(lines)


class LLMReformulator:
    """
    Rewrites queries using any LLM, with automatic strategy selection.

    Picks from: expand, narrow, decompose, step_back, hyde, perspective_shift
    based on why the previous evaluation failed.

    Args:
        generate_fn: A callable that accepts a prompt string and returns a
                     response string.

    Example::

        client = anthropic.Anthropic()

        def generate(prompt: str) -> str:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        reformulator = LLMReformulator(generate_fn=generate)
    """

    def __init__(self, generate_fn: Callable[[str], str]) -> None:
        self._generate = generate_fn

    def reformulate(
        self, original_query: str, history: list[LoopIteration]
    ) -> ReformulationOutput:
        history_summary = _build_history_summary(history)
        prompt = _PROMPT.format(query=original_query, history_summary=history_summary)
        raw = self._generate(prompt)

        try:
            data = json.loads(raw.strip())
            return ReformulationOutput(
                queries=list(data["queries"]),
                strategy=ReformulationStrategy(data["strategy"]),
                reasoning=str(data["reasoning"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            fallback_strategy = _select_strategy_heuristic(
                history[-1].evaluation.reason
            )
            return ReformulationOutput(
                queries=[original_query],
                strategy=fallback_strategy,
                reasoning=(
                    "LLM returned unparseable response; "
                    "fell back to heuristic strategy selection."
                ),
            )


class HeuristicReformulator:
    """
    Rewrites queries using keyword-based strategy selection — no LLM needed.

    Less accurate than LLMReformulator but costs nothing and has zero latency.
    Good for testing or cost-sensitive environments.
    """

    def reformulate(
        self, original_query: str, history: list[LoopIteration]
    ) -> ReformulationOutput:
        last = history[-1]
        strategy = _select_strategy_heuristic(last.evaluation.reason)

        if strategy == ReformulationStrategy.DECOMPOSE:
            parts = original_query.split(" and ")
            queries = [p.strip() for p in parts] if len(parts) > 1 else [original_query]
        else:
            queries = [original_query]

        return ReformulationOutput(
            queries=queries,
            strategy=strategy,
            reasoning=(
                f"Heuristic selected '{strategy}' based on: "
                f"{last.evaluation.reason[:100]}"
            ),
        )
