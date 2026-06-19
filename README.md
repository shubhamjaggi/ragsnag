# ragsnag

Loop engineering for RAG systems — iterative retrieval with automatic query reformulation.

[![CI](https://github.com/shubhamjaggi/ragsnag/actions/workflows/ci.yml/badge.svg)](https://github.com/shubhamjaggi/ragsnag/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragsnag)](https://pypi.org/project/ragsnag/)
[![Python](https://img.shields.io/pypi/pyversions/ragsnag)](https://pypi.org/project/ragsnag/)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/shubhamjaggi/ragsnag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The problem with standard RAG

Standard RAG is a one-shot pipeline:

```
query → retrieve 5 chunks → generate answer → done
```

If the retrieved chunks don't contain the answer — because the query used different vocabulary than your documents, because the question has multiple parts, because the relevant document is worded abstractly — the LLM either hallucinates or says "I don't know." You never find out that better chunks existed.

**ragsnag wraps that pipeline in a loop:**

```
query
 └─ retrieve chunks
      └─ generate answer
           └─ evaluate: is this grounded and complete?
                ├─ yes (score ≥ threshold) → return answer  [CONVERGED]
                └─ no  → reformulate query → retrieve again → ...
                                                          [MAX_ITERATIONS]
```

The loop keeps trying until the answer is good enough or the attempt limit is reached. Every iteration is recorded in a full trace so you can see exactly what happened and why.

---

## Install

```bash
pip install ragsnag
```

Requires Python 3.10+. The only dependency is `pydantic`. ragsnag has no opinion on which LLM or vector store you use.

---

## Quickstart

```python
from ragsnag import RAGLoop, LoopConfig, Chunk, GeneratorOutput, LLMEvaluator, LLMReformulator

# 1. Implement Retriever — fetches chunks from your vector store
class MyRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        results = my_pinecone_index.query(query, top_k=top_k)
        return [Chunk(content=r.text, source=r.id, score=r.score) for r in results]

# 2. Implement Generator — produces an answer from query + chunks
class MyGenerator:
    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
        context = "\n---\n".join(c.content for c in chunks)
        answer = my_llm(f"Answer using only this context:\n{context}\n\nQuestion: {query}")
        return GeneratorOutput(answer=answer, confidence=0.8)

# 3. Provide a generate_fn for the built-in evaluator and reformulator
def my_llm_fn(prompt: str) -> str:
    return my_llm(prompt)

# 4. Wire it up
loop = RAGLoop(
    retriever=MyRetriever(),
    generator=MyGenerator(),
    evaluator=LLMEvaluator(generate_fn=my_llm_fn),
    reformulator=LLMReformulator(generate_fn=my_llm_fn),
    config=LoopConfig(max_iterations=3, confidence_threshold=0.85),
)

result = loop.run("What is the refund policy for international orders?")

print(result.answer)        # the best answer found
print(result.stop_reason)   # StopReason.CONVERGED or MAX_ITERATIONS
print(result.iterations)    # how many attempts it took
```

See [`examples/`](examples/) for complete working code with Claude and OpenAI.

---

## How the loop works

### What stays constant across iterations

The **original user query** is passed to `generate()` and `evaluate()` on every iteration, regardless of what the Reformulator returns.

This is intentional. The Reformulator rewrites the query for **retrieval only** — to find better chunks. The Generator and Evaluator always work against the original question, because that is what the user actually asked. If the Generator received a reformulated query, it would answer a subtly different question.

```
Original query: "What is the refund policy for international orders?"

Iteration 2 reformulated retrieval query: "return eligibility cross-border shipping overseas"
                                           ↑ used by retrieve() only

Generator still receives: "What is the refund policy for international orders?"
Evaluator still receives: "What is the refund policy for international orders?"
```

### What changes across iterations

The **retrieval query** changes. After each failing iteration, the Reformulator analyzes the evaluation reason and writes a new query designed to fetch different, better chunks. This new query is used only for retrieval — it never reaches the Generator or Evaluator.

### How the best answer is chosen

ragsnag tracks the highest-scoring iteration throughout the loop. If the loop converges, the converging answer is returned. If `max_iterations` is hit without converging, the answer from the **highest-scoring iteration** is returned — not the last one.

This matters: if iteration 2 scores 0.85 and iteration 3 scores 0.60, you get iteration 2's answer even though iteration 3 ran last.

### Multi-query (DECOMPOSE strategy)

When the Reformulator's DECOMPOSE strategy returns multiple queries, ragsnag:

1. Calls `retrieve()` independently for each sub-query
2. Merges all returned chunks into one list
3. Deduplicates by exact content string (same chunk from two queries appears once)
4. Sorts the merged list by score descending
5. Passes the combined set to the Generator as context

The Generator still receives one combined context — it generates one answer. The loop logic is unchanged.

---

## API reference

### RAGLoop

```python
RAGLoop(
    retriever: Retriever,
    generator: Generator,
    evaluator: Evaluator,
    reformulator: Reformulator,
    config: LoopConfig | None = None,
)
```

#### `RAGLoop.run(query: str) → LoopResult`

Runs the loop for the given query. Blocks until convergence or max iterations. Returns a `LoopResult`.

---

### LoopConfig

```python
LoopConfig(
    max_iterations: int = 3,
    confidence_threshold: float = 0.8,
    top_k: int = 5,
    on_iteration: Callable[[LoopIteration], None] | None = None,
)
```

| Field | Default | Rationale |
|---|---|---|
| `max_iterations` | `3` | Each iteration costs retrieval + generate + evaluate. Beyond 3, marginal improvement typically shrinks while cost grows linearly. The loop exits early if the threshold is met, so raising this only raises the ceiling — it doesn't force more calls. |
| `confidence_threshold` | `0.8` | Conservative default. The Evaluator must rate the answer highly on both grounding and completeness. Lower to `0.65` if partial answers are acceptable. Raise to `0.95` for high-stakes Q&A where hallucinations are costly. |
| `top_k` | `5` | Chunks per retrieval call. More chunks = more context but higher cost and more noise. In DECOMPOSE mode, `top_k` applies per sub-query, so 2 sub-queries can produce up to `2 × top_k` chunks after deduplication. |
| `on_iteration` | `None` | Called after every iteration with the full `LoopIteration` object. Use for logging, progress indicators, or streaming partial results. |

**Tuning guide:** Start with the defaults. Look at `result.trace` after a few runs. If you see high scores on iteration 1 (≥0.9), lower `confidence_threshold`. If scores are rising slowly across iterations, increase `max_iterations`. If the first iteration is always good, you don't need the loop at all.

---

### LoopResult

```python
class LoopResult(BaseModel):
    answer: str
    confidence: float
    iterations: int
    stop_reason: StopReason
    trace: list[LoopIteration]

    @property
    def best_iteration(self) -> LoopIteration: ...
```

| Field | What it means |
|---|---|
| `answer` | The best answer found. If converged, this is the converging iteration's answer. If max iterations hit, this is the answer from the highest-scoring iteration — not necessarily the last one. |
| `confidence` | The Generator's self-reported confidence for the returned answer. Not the Evaluator's score. Use `result.best_iteration.evaluation.score` for the quality score. |
| `iterations` | How many loop iterations ran. Use this to measure cost — each iteration is roughly 2–3 LLM calls (generate + evaluate + possibly reformulate). |
| `stop_reason` | `CONVERGED` = found a good enough answer. `MAX_ITERATIONS` = gave up after the limit. Inspect the trace if you see `MAX_ITERATIONS`. |
| `trace` | Every `LoopIteration` in order. The primary tool for debugging. |
| `best_iteration` | The `LoopIteration` with the highest `evaluation.score`. When `stop_reason` is `CONVERGED`, this is also the last iteration. |

---

### LoopIteration

Every iteration in `result.trace` is a `LoopIteration`:

```python
class LoopIteration(BaseModel):
    iteration: int           # 1-indexed
    query: str               # retrieval query used (may be reformulated)
    chunks: list[Chunk]      # what was retrieved and passed to the generator
    answer: str              # what the generator produced
    confidence: float        # generator's self-reported confidence
    evaluation: EvaluationResult
```

```python
class EvaluationResult(BaseModel):
    is_grounded: bool   # answer supported by the chunks?
    is_complete: bool   # question fully answered?
    score: float        # 0.0–1.0 overall quality
    reason: str         # why this score — also used by the Reformulator
```

---

### StopReason

| Value | Meaning |
|---|---|
| `CONVERGED` | Evaluation score ≥ threshold. Loop found a good answer. |
| `MAX_ITERATIONS` | Ran out of attempts. Inspect `result.trace` to see why scores stayed low. |
| `HUMAN_APPROVED` | Reserved for future human-in-the-loop workflows. |
| `ERROR` | Reserved for future structured error handling. Exceptions currently propagate directly. |

---

## Built-in components

### LLMEvaluator

Evaluates answer quality using any LLM via a simple callable.

```python
from ragsnag import LLMEvaluator

evaluator = LLMEvaluator(generate_fn=my_llm_fn)
# generate_fn: Callable[[str], str]
# Takes a prompt string, returns a response string.
```

**How it works:** Sends the original query, all chunk contents, and the generated answer to the LLM in a structured prompt. Asks the LLM to respond with JSON containing `is_grounded`, `is_complete`, `score`, and `reason`. Parses the JSON and returns an `EvaluationResult`.

**Error handling:** If the LLM returns unparseable JSON or is missing required fields, the evaluator returns `score=0.0` with `is_grounded=False` and a reason string explaining what went wrong. The loop continues rather than crashing — the next iteration will try to do better.

**When to use:** Use `LLMEvaluator` as your default. It produces accurate scores and — crucially — writes high-quality `reason` strings that the Reformulator uses to select the right strategy.

---

### LLMReformulator

Rewrites the retrieval query using any LLM, with automatic strategy selection.

```python
from ragsnag import LLMReformulator

reformulator = LLMReformulator(generate_fn=my_llm_fn)
```

**How it works:** Sends the original query and a summary of all previous iterations (queries tried, answers generated, evaluation scores, reasons) to the LLM. Asks it to pick a reformulation strategy and write new queries. Expects JSON with `strategy`, `queries`, and `reasoning`.

**Error handling:** If the LLM returns unparseable JSON or an invalid strategy, the reformulator falls back to `HeuristicReformulator` using the last iteration's reason string. The loop never crashes due to a bad reformulation response.

**When to use:** Use `LLMReformulator` in production. It selects strategies more accurately than the heuristic version because it can reason about the full iteration history, not just keyword matching.

---

### HeuristicReformulator

Selects a reformulation strategy by matching keywords in the evaluator's reason string. No LLM required.

```python
from ragsnag import HeuristicReformulator

reformulator = HeuristicReformulator()
```

**How it works:** Checks the last iteration's `evaluation.reason` for keywords associated with each strategy:

| Keywords in reason | Strategy selected |
|---|---|
| "vocabulary", "terminology", "no chunk", "not retrieved", "jargon" | `PERSPECTIVE_SHIFT` |
| "too generic", "too broad", "surface level", "vague", "incomplete" | `NARROW` |
| "multi-part", "multiple question", "missing second", "also ask" | `DECOMPOSE` |
| "no relevant", "completely off", "unrelated", "abstract" | `STEP_BACK` |
| "conceptual", "how does", "explain", "mechanism" | `HYDE` |
| anything else | `EXPAND` (default) |

**When to use:** During development and testing (avoids LLM costs), in cost-sensitive production environments, or as a fallback inside `LLMReformulator` (already done automatically).

**Limitation:** Keyword matching is brittle. The strategy selected depends entirely on how your `Evaluator` phrases its reason strings. If your evaluator writes generic reasons ("answer is incomplete"), the heuristic will default to `EXPAND` regardless of the real problem. `LLMReformulator` doesn't have this limitation.

---

## Reformulation strategies

When the Evaluator's score is below threshold, the Reformulator picks one of these strategies:

| Strategy | Root cause it targets | What it does | Example |
|---|---|---|---|
| `EXPAND` | Query too narrow | Adds synonyms and related terms | `"refund policy"` → `"refund return reimbursement money-back policy"` |
| `NARROW` | Results too broad or generic | Makes the query more specific | `"employee benefits"` → `"parental leave weeks paid maternity"` |
| `DECOMPOSE` | Multi-part question | Splits into sub-queries, merges chunk results | `"pricing and billing"` → `["pricing tiers", "billing cycle annual monthly"]` |
| `STEP_BACK` | Nothing relevant found | Asks a broader question first | `"max upload size free tier mobile"` → `"file upload size limits"` |
| `HYDE` | Abstract or conceptual question | Searches with a hypothetical answer instead of the question | Query: `"what causes API latency spikes"` → searches for `"latency spikes are caused by connection pool exhaustion..."` |
| `PERSPECTIVE_SHIFT` | Vocabulary mismatch | Rewrites using different terminology | `"how to cancel"` → `"account termination close subscription"` |

**Why HYDE works:** Your documents are written as answers ("The maximum upload size is 50MB"), not as questions. Embedding a question and an answer-shaped sentence into the same vector space produces a better similarity match than question-to-question.

---

## Debugging with the trace

When `stop_reason` is `MAX_ITERATIONS`, the trace tells you why:

```python
result = loop.run("What is the late payment penalty on overdue invoices?")

if result.stop_reason == StopReason.MAX_ITERATIONS:
    for it in result.trace:
        print(f"\n--- Iteration {it.iteration} ---")
        print(f"Query:      {it.query}")
        print(f"Score:      {it.evaluation.score:.2f}")
        print(f"Grounded:   {it.evaluation.is_grounded}")
        print(f"Complete:   {it.evaluation.is_complete}")
        print(f"Reason:     {it.evaluation.reason}")
        print(f"Top chunk:  {it.chunks[0].content[:100] if it.chunks else 'none'}")
```

**Common patterns and what to do:**

| What you see in the trace | Likely cause | Fix |
|---|---|---|
| Score stays at 0.0 every iteration | Evaluator failing to parse LLM response | Check `reason` field — it contains the raw response. Fix the LLM prompt. |
| Score is 0.3–0.5, `is_grounded=False` | LLM ignoring context, answering from memory | Strengthen the generator prompt: "Answer ONLY from the context below." |
| Score is 0.5–0.7, `is_complete=False` | Partial answer, chunks missing key info | Increase `top_k`, try `DECOMPOSE` strategy manually, or add more documents. |
| Score is high (0.8+) but loop doesn't stop | `confidence_threshold` set too high | Lower the threshold, or check that your evaluator's scoring scale matches it. |
| Chunks look unrelated on every iteration | Vocabulary mismatch or wrong documents indexed | Check the `chunks` field in iteration 1. If scores are all < 0.5, the issue is in the index, not the loop. |

---

## Writing custom components

### Custom Evaluator

Implement the `Evaluator` protocol — any class with an `evaluate` method:

```python
from ragsnag import Chunk, EvaluationResult

class KeywordEvaluator:
    """Simple keyword overlap evaluator. No LLM required."""

    def evaluate(
        self, query: str, chunks: list[Chunk], answer: str
    ) -> EvaluationResult:
        context_words = set(" ".join(c.content for c in chunks).lower().split())
        answer_words = set(answer.lower().split()) - {"the", "a", "an", "is", "in"}

        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        is_grounded = overlap >= 0.5
        is_complete = len(answer.split()) >= 15  # crude completeness proxy

        return EvaluationResult(
            is_grounded=is_grounded,
            is_complete=is_complete,
            score=round(overlap, 2),
            reason=f"Keyword overlap {overlap:.0%}. {'Grounded.' if is_grounded else 'Possible hallucination.'} "
                   f"{'Complete.' if is_complete else 'Answer may be too short.'}",
        )
```

**Important:** Write specific, informative `reason` strings. The Reformulator reads them to decide which strategy to apply. "too generic results, no mention of international" is far more useful than "incomplete".

### Custom Reformulator

```python
from ragsnag import LoopIteration, ReformulationOutput, ReformulationStrategy

class DomainReformulator:
    """Expands queries using domain-specific synonym lists."""

    _SYNONYMS = {
        "refund":        ["return", "reimbursement", "money back"],
        "cancel":        ["terminate", "close account", "end subscription"],
        "international": ["cross-border", "overseas", "global", "foreign"],
    }

    def reformulate(
        self, original_query: str, history: list[LoopIteration]
    ) -> ReformulationOutput:
        extra = []
        for word in original_query.lower().split():
            extra.extend(self._SYNONYMS.get(word, []))

        new_query = f"{original_query} {' '.join(extra)}" if extra else original_query
        return ReformulationOutput(
            queries=[new_query],
            strategy=ReformulationStrategy.EXPAND,
            reasoning=f"Appended domain synonyms: {extra or 'none found'}",
        )
```

### Custom Retriever

```python
from ragsnag import Chunk

class ChromaRetriever:
    def __init__(self, collection, embed_fn) -> None:
        self._collection = collection
        self._embed = embed_fn

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        embedding = self._embed(query)
        results = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        return [
            Chunk(
                content=doc,
                source=meta.get("source", "unknown"),
                score=1.0 - dist,  # Chroma returns distances, convert to similarity
                metadata=meta,
            )
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
```

---

## Using the trace for observability

`LoopResult` is a Pydantic model — you can serialize it for logging or storage:

```python
import json

result = loop.run("What is the refund policy?")

# Serialize to dict (suitable for JSON logging, databases, etc.)
data = result.model_dump()

# Log as JSON
print(json.dumps(data, indent=2, default=str))

# Access best iteration
best = result.best_iteration
print(f"Best answer came from iteration {best.iteration} with score {best.evaluation.score}")
print(f"Query used in that iteration: {best.query}")
print(f"Why that iteration wasn't the final: {best.evaluation.reason}")
```

Stream progress with `on_iteration`:

```python
def log_iteration(it: LoopIteration) -> None:
    print(f"[{it.iteration}] score={it.evaluation.score:.2f} | {it.evaluation.reason}")

loop = RAGLoop(
    ...,
    config=LoopConfig(on_iteration=log_iteration),
)
```

---

## What ragsnag does NOT do

- **No document ingestion or chunking.** ragsnag does not split documents, generate embeddings, or manage indexes. That is your Retriever's job.
- **No vector database management.** Bring your own Pinecone, Chroma, Weaviate, or any other store.
- **No LLM client.** ragsnag takes a `Callable[[str], str]` for the built-in evaluator and reformulator. You wire up the LLM client yourself.
- **No prompt templates baked in.** Your Generator owns its prompt. ragsnag only defines what goes in (query + chunks) and what comes out (answer + confidence).
- **No vendor lock-in.** Every component is a protocol. If Pinecone shuts down tomorrow, you swap the Retriever — nothing else changes.

---

## Development

```bash
git clone https://github.com/shubhamjaggi/ragsnag
cd ragsnag
pip install -e ".[dev]"

# Run tests with coverage
pytest tests/ -v

# Lint
ruff check src/

# Type check
mypy src/ragsnag

# Install pre-commit hooks
pip install pre-commit && pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## License

MIT — see [LICENSE](LICENSE).
