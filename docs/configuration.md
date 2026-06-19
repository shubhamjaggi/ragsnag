# Configuration reference

Complete reference for all ragsnag configuration, with rationale and tuning guidance.

---

## LoopConfig

```python
from ragsnag import LoopConfig

config = LoopConfig(
    max_iterations=3,
    confidence_threshold=0.8,
    top_k=5,
    on_iteration=None,
)
```

---

### `max_iterations: int = 3`

Maximum number of retrieve → generate → evaluate cycles before the loop gives up.

**Why this exists:** Each iteration costs money and adds latency. The loop must have a hard ceiling so it doesn't run indefinitely if the Evaluator never scores high enough.

**What it controls:** The ceiling on attempts. The loop exits early (CONVERGED) if `confidence_threshold` is met, so raising `max_iterations` does not mean more LLM calls in the happy path — it only changes the worst case.

**Cost model:** Each iteration is approximately:
- 1 retrieval call (fast, usually cheap)
- 1 LLM call for generation
- 1 LLM call for evaluation
- 1 LLM call for reformulation (except the last iteration)

So `max_iterations=3` costs at most 3 retrievals + 3 generates + 3 evaluates + 2 reformulations = 8 LLM calls in the worst case.

**When to increase:** If you look at traces and see scores still rising at iteration 3 (e.g., 0.5 → 0.65 → 0.73 → would have converged at 4), increase to 4 or 5.

**When to decrease:** If traces show convergence almost always at iteration 1, set `max_iterations=1` and skip the overhead. This effectively degrades ragsnag to standard one-shot RAG, which may be what you want for fast, cheap endpoints.

**Valid range:** `>= 1`. Setting to 1 means no reformulation ever happens (the single iteration either converges or returns the best of one attempt).

---

### `confidence_threshold: float = 0.8`

The minimum evaluation score required to stop the loop and return CONVERGED.

**Why this exists:** Without a threshold, the loop would always run all `max_iterations` regardless of answer quality.

**What it controls:** The definition of "good enough." The Evaluator returns a score from 0.0 to 1.0. When that score meets or exceeds this threshold, the loop stops immediately.

**The threshold must match your evaluator's scoring scale.** If your Evaluator gives `0.9` for a mediocre answer, a threshold of `0.8` will accept mediocre answers. If your Evaluator gives `0.5` for good answers, a threshold of `0.8` will never converge.

**Tuning guidance:**

| Use case | Recommended threshold |
|---|---|
| Internal search, drafting, low-stakes Q&A | 0.65–0.70 |
| Customer-facing Q&A, documentation assistant | 0.80–0.85 (default) |
| Legal, financial, medical, compliance | 0.90–0.95 |
| Never stop early (always use all iterations) | 1.01 (unreachable) |

**Valid range:** `0.0 – 1.0` inclusive. Setting to `0.0` means the first iteration always converges regardless of quality. Setting to `1.0` means only a perfect score converges.

---

### `top_k: int = 5`

How many chunks to retrieve per query per iteration.

**Why this exists:** You need to tell ragsnag how many chunks to request from the Retriever. ragsnag passes this value directly to `retriever.retrieve(query, top_k=top_k)`.

**What it controls:** The size of the context window that the Generator receives. More chunks = more context = potentially better answers, but also higher cost (longer prompts), higher latency, and more noise from irrelevant chunks.

**In DECOMPOSE mode:** `top_k` is applied per sub-query. If DECOMPOSE returns 2 queries and `top_k=5`, you retrieve up to 5 chunks per query = up to 10 chunks in the merged set (after deduplication). Plan your Generator's context window accordingly.

**Tuning guidance:**

| Scenario | Recommended top_k |
|---|---|
| Short, dense documents (FAQs, policies) | 3–5 |
| Long documents (contracts, reports) | 5–10 |
| Many small chunks (< 100 words each) | 8–15 |
| Very large corpus, high recall needed | 10–20 |
| Cost-sensitive, fast responses needed | 3 |

**Valid range:** `>= 1`.

---

### `on_iteration: Callable[[LoopIteration], None] | None = None`

An optional callback called after every iteration, immediately after the Evaluator returns.

**Why this exists:** ragsnag is a library, not a server. It has no built-in logging, metrics, or UI. The callback gives you a hook to add whatever observability you need without ragsnag having to know about your logging infrastructure.

**When it is called:** After `history.append(iteration)`, before the stop-or-reformulate decision. This means you see every iteration, including the one that causes convergence.

**What you receive:** The full `LoopIteration` object — query used, chunks retrieved, answer generated, confidence, and evaluation result.

**Common uses:**

```python
# Logging
on_iteration=lambda it: logger.info(
    "ragsnag iter %d score=%.2f reason=%s",
    it.iteration, it.evaluation.score, it.evaluation.reason
)

# Metrics / monitoring
on_iteration=lambda it: metrics.gauge(
    "ragsnag.score", it.evaluation.score, tags={"iter": it.iteration}
)

# Progress display
on_iteration=lambda it: print(
    f"  [{it.iteration}] {it.evaluation.score:.0%} — {it.evaluation.reason}"
)

# Streaming partial results to a caller
results_queue = Queue()
on_iteration=lambda it: results_queue.put(it.answer) if it.evaluation.score > 0.5 else None

# Abort on specific condition (raise to exit the loop early)
on_iteration=lambda it: (_ for _ in ()).throw(
    RuntimeError("aborted")
) if it.iteration > 1 and it.evaluation.score < 0.2 else None
```

**Valid values:** Any callable with signature `(LoopIteration) -> None`, or `None`.

---

## Choosing between LLMEvaluator and a custom Evaluator

**Use `LLMEvaluator` when:**
- You want accurate evaluation without writing custom logic
- You want high-quality `reason` strings that produce better reformulations
- You don't mind the extra LLM call per iteration

**Write a custom Evaluator when:**
- You want zero-cost evaluation (heuristic, keyword-based)
- You have domain-specific rules for what counts as "grounded" or "complete"
- You want to combine multiple signals (keyword overlap + LLM confidence + length check)
- You're in a high-throughput setting where an extra LLM call per iteration is too expensive

**The most important thing your Evaluator does:** Write specific `reason` strings. The `LLMReformulator` and `HeuristicReformulator` read `reason` to choose the next strategy. "chunks describe domestic policy only; question asks about international orders" leads to a precise `PERSPECTIVE_SHIFT` or `NARROW` reformulation. "answer is incomplete" leads to a generic `EXPAND`.

---

## Choosing between LLMReformulator and HeuristicReformulator

**Use `LLMReformulator` when:**
- You want the best reformulation quality
- You want the reformulator to reason about the full iteration history (not just the last reason)
- You're already paying for LLM calls in the evaluator, so one more call per iteration is marginal cost

**Use `HeuristicReformulator` when:**
- You're testing or prototyping (no LLM cost)
- Your Evaluator writes consistent, keyword-rich reason strings that the heuristic can match
- Latency is critical and you want to eliminate one LLM call per iteration
- You're in a cost-sensitive environment

**Note:** `LLMReformulator` automatically falls back to `HeuristicReformulator` if the LLM returns unparseable output. You don't need to wire this up yourself.

---

## Tuning workflow

1. Run your loop on 10–20 representative queries with `max_iterations=5` and `confidence_threshold=0.99` (so it always runs all iterations).
2. Examine `result.trace` for each run:
   - What iteration did the score first reach what you'd consider "acceptable"?
   - Did scores improve monotonically or oscillate?
   - Were there iterations where the reformulation made things worse?
3. Set `max_iterations` to the 90th percentile iteration count needed (e.g., if 90% of queries converge by iteration 3, set `max_iterations=3`).
4. Set `confidence_threshold` to the score you observed for answers you'd actually ship.
5. Consider setting `top_k` higher if `is_complete=False` dominates your traces (the answer exists but wasn't retrieved).
