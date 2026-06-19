# How ragsnag works

This document explains the internals of the loop engine — what runs, in what order, what each component is responsible for, and the design decisions behind each choice.

---

## The loop lifecycle

When you call `RAGLoop.run(query)`, the following sequence runs up to `max_iterations` times:

```
┌─────────────────────────────────────────────────────────────┐
│  RAGLoop.run(query)                                         │
│                                                             │
│  current_queries = [query]          ← starts as user query │
│  history = []                                               │
│                                                             │
│  for i in range(max_iterations):                           │
│    ┌──────────────────────────────────────────────────┐    │
│    │ 1. RETRIEVE                                      │    │
│    │    chunks = retrieve(current_queries, top_k)     │    │
│    │    (merge + deduplicate if multiple queries)     │    │
│    └────────────────────┬─────────────────────────────┘    │
│                         ↓                                   │
│    ┌──────────────────────────────────────────────────┐    │
│    │ 2. GENERATE                                      │    │
│    │    output = generator.generate(query, chunks)   │    │
│    │    ← always original query, not reformulated    │    │
│    └────────────────────┬─────────────────────────────┘    │
│                         ↓                                   │
│    ┌──────────────────────────────────────────────────┐    │
│    │ 3. EVALUATE                                      │    │
│    │    result = evaluator.evaluate(query, chunks,    │    │
│    │                                output.answer)   │    │
│    │    ← always original query, not reformulated    │    │
│    └────────────────────┬─────────────────────────────┘    │
│                         ↓                                   │
│    ┌──────────────────────────────────────────────────┐    │
│    │ 4. RECORD + CALLBACK                             │    │
│    │    append LoopIteration to history               │    │
│    │    call on_iteration(iteration) if set           │    │
│    │    update best if score improved                 │    │
│    └────────────────────┬─────────────────────────────┘    │
│                         ↓                                   │
│    ┌──────────────────────────────────────────────────┐    │
│    │ 5. STOP CHECK                                    │    │
│    │    if score >= confidence_threshold:             │    │
│    │        return LoopResult(CONVERGED)              │    │
│    └────────────────────┬─────────────────────────────┘    │
│                         ↓                                   │
│    ┌──────────────────────────────────────────────────┐    │
│    │ 6. REFORMULATE (skipped on last iteration)       │    │
│    │    output = reformulator.reformulate(            │    │
│    │        query, history)                           │    │
│    │    current_queries = output.queries              │    │
│    └────────────────────┬─────────────────────────────┘    │
│                         ↓ (next iteration)                  │
│                                                             │
│  return LoopResult(MAX_ITERATIONS, answer=best.answer)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Why the original query never changes

The original query is passed to `generate()` and `evaluate()` on every iteration, regardless of what the Reformulator returns.

**Why:** Reformulation changes the *retrieval* query — the string used to search the vector index. It does not change the user's question. If the Generator received a reformulated query like "return eligibility cross-border shipment" instead of "What is the refund policy for international orders?", it would produce a subtly different answer shaped around retrieval jargon rather than the user's intent. Similarly, the Evaluator must always judge: "Does this answer the user's original question?" — not "Does this answer the reformulated retrieval query?"

**Practical effect:** This means your Generator and Evaluator implementations will always see the user's natural-language question. You don't need to handle the case where the query changes between iterations.

---

## Why the best answer is returned, not the last

When `max_iterations` is reached without converging, ragsnag returns the answer from the highest-scoring iteration — not iteration N.

**Why:** Later iterations are not guaranteed to be better. The Reformulator can misread the evaluation reason and produce a query that retrieves worse chunks. If iteration 2 scores 0.85 and iteration 3 scores 0.50, returning iteration 3's answer would be strictly worse.

ragsnag tracks `best` continuously and replaces it whenever a higher score is seen. This is a simple max-tracking algorithm with no lookahead.

---

## How chunk deduplication works

When the DECOMPOSE strategy returns multiple sub-queries, `_retrieve_and_merge` runs `retrieve()` for each query independently, then:

1. Iterates through all results in the order they were returned
2. Checks if the chunk's `content` string has been seen before (exact string match)
3. If not seen: adds it to the result list and marks the content as seen
4. Sorts the final list by `score` descending

**Why exact string deduplication:** The same document chunk will have the same `content` string regardless of which query retrieved it. Using `content` as the deduplication key is reliable and doesn't require comparing `source` IDs (which might differ for the same text).

**Why sort after merge:** Each retriever call returns chunks sorted by score within that query. After merging two or more query results, the relative ordering is no longer guaranteed. A chunk from query B might score higher than the top chunk from query A.

---

## Why reformulation is skipped on the last iteration

The Reformulator is called after every iteration except the last one.

**Why:** The reformulated query would only be used in the next iteration. If there is no next iteration, the reformulation is wasted — it costs an LLM call and produces nothing. The code explicitly checks `if i < max_iterations - 1` before calling the Reformulator.

---

## How strategy selection works in LLMReformulator

`LLMReformulator` sends the full iteration history (all queries tried, all evaluation reasons, all scores) to the LLM and asks it to:

1. Pick a strategy from the six options
2. Write one or more new queries
3. Explain why it chose that strategy

The LLM response must be valid JSON with `strategy`, `queries`, and `reasoning` fields. If the response is not valid JSON or contains an unknown strategy value, `LLMReformulator` catches the error and falls back to `_select_strategy_heuristic(last_reason)` — the same module-level keyword-matching function used by `HeuristicReformulator`.

**Why fall back to heuristic instead of crashing:** The loop should be robust to LLM failures. A bad reformulation is worse than a good one but better than an exception. The loop continues and the next iteration may produce a better result.

---

## How HeuristicReformulator selects strategies

`_select_strategy_heuristic(reason)` checks the evaluation reason string for keywords associated with each strategy, in priority order:

1. `PERSPECTIVE_SHIFT` — vocabulary/terminology mismatch, no chunks found, nothing retrieved
2. `NARROW` — too generic, too broad, vague, surface level, incomplete, or missing specifics
3. `DECOMPOSE` — multi-part question, missing second part, multiple questions
4. `STEP_BACK` — no relevant results, completely off-topic, unrelated, or abstract
5. `HYDE` — conceptual question, "how does", explain, mechanism, describe
6. `EXPAND` — default (no specific keyword matched)

The first matching strategy wins. This is intentional: vocabulary mismatch is the most specific diagnosis and should take priority over the generic "expand" fallback.

---

## How LLMEvaluator works

`LLMEvaluator` sends a structured prompt to the LLM containing:
- The original query
- All chunk contents, separated by `---`
- The generated answer

It asks the LLM to respond with JSON containing `is_grounded`, `is_complete`, `score`, and `reason`.

**Why JSON output:** Structured output is easier to parse reliably than free-text. The prompt explicitly instructs the LLM to return only JSON with no surrounding markdown or explanation.

**On parse failure:** If the JSON is malformed or missing fields, `LLMEvaluator` returns `score=0.0` with `is_grounded=False` and a `reason` that includes the raw response text. This means the Reformulator sees a reason like "Evaluator returned unparseable response: ..." and can potentially adapt. The loop does not crash.

---

## What the trace is for

`LoopResult.trace` is a list of `LoopIteration` objects — one per iteration, in order. Each iteration records:

- The query used for retrieval (reformulated or original)
- All chunks that were retrieved
- The answer that was generated
- The full evaluation result (score, grounding, completeness, reason)

The trace exists because debugging RAG failures is hard without it. Without a trace, you see the final answer and nothing else. With a trace, you can answer:

- Did the first retrieval fetch relevant chunks?
- Did the score improve across iterations, or stay flat?
- Which strategy was used and did it help?
- Was the answer grounded in the chunks or did the LLM hallucinate?
- What did the Evaluator say that caused the next reformulation?

`LoopResult` is a Pydantic model — call `.model_dump()` to serialize the entire trace to a dict, or `.model_dump_json()` for JSON.
