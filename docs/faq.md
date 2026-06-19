# FAQ

---

**Q: Do I have to use an LLM for the Evaluator and Reformulator?**

No. `HeuristicReformulator` uses keyword matching with no LLM. You can also write a completely custom Evaluator that uses heuristics (keyword overlap, answer length, etc.) with no LLM calls at all. The tradeoff: heuristic components are cheaper and faster but less accurate. `HeuristicReformulator` depends entirely on well-worded reason strings from your Evaluator to select the right strategy.

---

**Q: Why does the loop sometimes return an answer from iteration 2 instead of iteration 3?**

ragsnag returns the answer with the highest evaluation score, not the last answer. If iteration 2 scored 0.85 and iteration 3 scored 0.70, the iteration 2 answer is returned. This prevents regressions — a bad reformulation in a later iteration can produce worse chunks and a worse answer.

---

**Q: What happens if the Evaluator returns unparseable output?**

`LLMEvaluator` catches JSON parse errors and returns `score=0.0` with a `reason` string that includes the raw LLM response. The loop continues. The low score triggers reformulation, and the next iteration may recover. The loop never crashes due to an evaluator failure.

---

**Q: What happens if the Reformulator returns unparseable output?**

`LLMReformulator` catches parse errors and falls back to `_select_strategy_heuristic(last_reason)` — the module-level keyword-matching function shared with `HeuristicReformulator`. The fallback returns the original query with a heuristic-selected strategy. The loop continues. Again, no crash.

---

**Q: What does `confidence` in `GeneratorOutput` control?**

Nothing in the loop. It is stored in `LoopIteration.confidence` for observability — you can inspect it in the trace — but the loop control decision uses only `EvaluationResult.score`. The evaluator's score is what matters. The generator's confidence is self-reported and unverified.

---

**Q: Why is the reformulated query not passed to the Generator?**

The reformulated query is a retrieval artifact — it exists to find better chunks, not to answer a different question. If the Generator received "return eligibility cross-border shipment" instead of "What is the refund policy for international orders?", it would produce an answer shaped around retrieval jargon rather than the user's original intent. The original query is always the question being answered.

---

**Q: My scores are always 0.0. What's wrong?**

`LLMEvaluator` returns `score=0.0` when it can't parse the LLM's response. Check `result.trace[0].evaluation.reason` — if it starts with "Evaluator returned unparseable response", your LLM is not returning valid JSON. Possible causes:
- The LLM is wrapping the JSON in markdown code fences (```json ... ```)
- The LLM is adding a preamble before the JSON
- The model you're using doesn't follow JSON instructions well

Fix: try a more instruction-following model, or add explicit instructions like "Return ONLY the JSON object. No markdown, no explanation, no preamble."

---

**Q: The loop always hits MAX_ITERATIONS. How do I debug this?**

Print the trace:

```python
result = loop.run("your query")
for it in result.trace:
    print(f"Iter {it.iteration}: score={it.evaluation.score:.2f} | {it.evaluation.reason}")
    if it.chunks:
        print(f"  Top chunk: {it.chunks[0].content[:100]}")
    else:
        print("  No chunks retrieved")
```

Common causes:
- **Scores stuck at 0.0**: evaluator failing, see above.
- **Scores at 0.3–0.5, not improving**: the right documents don't exist in your index. No amount of reformulation will help — improve your data.
- **Scores improving but not reaching threshold**: lower `confidence_threshold`, or increase `max_iterations`.
- **No chunks retrieved**: your retriever is returning empty results. The query vocabulary doesn't match your index.

---

**Q: Can I run multiple queries in parallel?**

`RAGLoop` is stateless — `run()` has no shared state between calls. You can safely call `loop.run()` from multiple threads or coroutines simultaneously. Each call maintains its own `history` list and `best` tracker.

---

**Q: How do I serialize the result for logging or storage?**

`LoopResult` is a Pydantic model:

```python
result = loop.run("query")

# Dict
data = result.model_dump()

# JSON string
json_str = result.model_dump_json()

# Exclude the full chunk content to save space
compact = result.model_dump(exclude={"trace": {"__all__": {"chunks": {"__all__": {"content"}}}}})
```

---

**Q: How do I add ragsnag to an existing RAG pipeline without rewriting everything?**

Wrap your existing retriever and generator in thin adapter classes:

```python
# Existing code
def my_existing_retriever(query: str) -> list[dict]:
    return vector_db.search(query, k=5)

def my_existing_generator(query: str, docs: list[str]) -> str:
    return llm.complete(build_prompt(query, docs))

# ragsnag adapters — minimal wrappers
class RetrieverAdapter:
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        results = my_existing_retriever(query)
        return [Chunk(content=r["text"], source=r["id"], score=r["score"]) for r in results[:top_k]]

class GeneratorAdapter:
    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
        docs = [c.content for c in chunks]
        answer = my_existing_generator(query, docs)
        return GeneratorOutput(answer=answer, confidence=0.8)
```

---

**Q: What is `StopReason.HUMAN_APPROVED`? How do I use it?**

It is reserved for future human-in-the-loop workflows where a human reviews the answer mid-loop and approves or rejects it. It is not currently triggered by the built-in loop logic. You can trigger it from an `on_iteration` callback by raising a custom exception and catching it outside `loop.run()`, then constructing a `LoopResult` manually with `StopReason.HUMAN_APPROVED`.

---

**Q: Does ragsnag support async?**

Not in v0.1. All protocols are synchronous. If your LLM client is async, wrap the async call in `asyncio.run()` inside your `generate_fn`, or use a thread executor. Async support is on the roadmap and will be added when there is sufficient user demand for it.

---

**Q: How do I publish a new release?**

Create and push a version tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The `publish.yml` GitHub Actions workflow detects the tag and publishes to PyPI automatically using OIDC trusted publishing. You need to configure the PyPI trusted publisher for the `shubhamjaggi/ragsnag` repository once in your PyPI project settings.
