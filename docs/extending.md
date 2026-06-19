# Extending ragsnag

ragsnag is built on four protocols. Any class that implements the right method signature satisfies the protocol — no inheritance or registration required. Python's `isinstance(obj, Protocol)` check confirms conformance at runtime if needed.

---

## The four protocols

```python
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]: ...

class Generator(Protocol):
    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput: ...

class Evaluator(Protocol):
    def evaluate(self, query: str, chunks: list[Chunk], answer: str) -> EvaluationResult: ...

class Reformulator(Protocol):
    def reformulate(self, original_query: str, history: list[LoopIteration]) -> ReformulationOutput: ...
```

Implement whichever you need. Use `LLMEvaluator` and `LLMReformulator` for the ones you don't want to write yourself.

---

## Writing a Retriever

Your Retriever wraps your vector database. The only requirement is returning a list of `Chunk` objects with scores in `[0.0, 1.0]`.

```python
from ragsnag import Chunk

class PineconeRetriever:
    def __init__(self, index, embed_fn) -> None:
        self._index = index
        self._embed = embed_fn

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        vector = self._embed(query)
        results = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            Chunk(
                content=match.metadata["text"],
                source=match.id,
                score=match.score,          # Pinecone cosine scores are in [0, 1]
                metadata={"namespace": match.namespace},
            )
            for match in results.matches
        ]
```

**Score normalization:** Some vector stores return raw dot products or distances rather than [0, 1] similarity scores. Normalize before constructing `Chunk`:

```python
# Chroma returns distances (lower = more similar). Convert to similarity:
score = 1.0 - distance  # only valid if distances are in [0, 1]

# Elasticsearch BM25 returns unbounded scores. Normalize:
max_score = max(hit.score for hit in hits) or 1.0
score = hit.score / max_score
```

**Empty results:** If the vector store returns nothing, return an empty list `[]`. ragsnag will pass an empty chunk list to the Generator, which should then produce a low-confidence answer that the Evaluator will score low, triggering reformulation.

---

## Writing a Generator

Your Generator produces an answer from the original query and the retrieved chunks.

```python
from ragsnag import Chunk, GeneratorOutput

class MyGenerator:
    _PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer concisely and precisely. End your response with:
CONFIDENCE: <a number from 0.0 to 1.0 reflecting how sure you are>"""

    def generate(self, query: str, chunks: list[Chunk]) -> GeneratorOutput:
        if not chunks:
            return GeneratorOutput(answer="No relevant documents were found.", confidence=0.0)

        context = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.content}" for c in chunks
        )
        raw = self._call_llm(self._PROMPT.format(context=context, query=query))
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> GeneratorOutput:
        if "CONFIDENCE:" in raw:
            parts = raw.rsplit("CONFIDENCE:", 1)
            answer = parts[0].strip()
            try:
                confidence = float(parts[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.7
        else:
            answer = raw.strip()
            confidence = 0.7
        return GeneratorOutput(answer=answer, confidence=confidence)

    def _call_llm(self, prompt: str) -> str:
        raise NotImplementedError
```

**Key points:**
- Always receive the ORIGINAL user query, not the reformulated one. Do not try to detect or use the reformulated query.
- Include all chunk contents in the prompt context. ragsnag has already sorted them by score (most relevant first), so you can truncate from the end if context is too long.
- Return `confidence` honestly. If you ask the LLM to self-report, pass that through. If you use a fixed default, use something like `0.7` — it is stored in the trace for observability but does not affect loop control.
- Handle empty chunks: if no chunks were retrieved, the Generator should reflect that uncertainty in both the answer and a low confidence score.

---

## Writing an Evaluator

Your Evaluator scores how well the generated answer addresses the original question given the retrieved chunks.

```python
from ragsnag import Chunk, EvaluationResult

class MyEvaluator:
    def evaluate(
        self, query: str, chunks: list[Chunk], answer: str
    ) -> EvaluationResult:
        # ... scoring logic ...
        return EvaluationResult(
            is_grounded=True,
            is_complete=False,
            score=0.55,
            reason="Answer is factually supported by the chunks but only addresses "
                   "the domestic policy. The question asks specifically about "
                   "international orders, which no retrieved chunk covers.",
        )
```

**What makes a good `reason` string:**

The `reason` is the single most important output your Evaluator produces. The Reformulator reads it to select the next strategy. Write reasons that are:

- **Specific about what is missing:** "No chunk mentions international shipping" is better than "incomplete"
- **Descriptive of the gap:** "chunks describe pricing but not the billing cycle" tells the Reformulator to try DECOMPOSE or NARROW
- **Vocabulary-aware:** "The document uses 'account closure' but the query says 'cancel'" tells the Reformulator to try PERSPECTIVE_SHIFT

**Scoring consistency:** Pick a scoring convention and stick to it. The `confidence_threshold` in `LoopConfig` must be calibrated to your score distribution. A rough guide:

| Score | Meaning |
|---|---|
| 0.9–1.0 | Answer is grounded, complete, and accurate |
| 0.7–0.9 | Mostly good — minor gaps or minor grounding issues |
| 0.5–0.7 | Partial answer or partially grounded |
| 0.3–0.5 | Answer exists but has significant issues |
| 0.0–0.3 | Wrong, hallucinated, or evaluator failed |

---

## Writing a Reformulator

Your Reformulator produces new retrieval queries based on what went wrong.

```python
from ragsnag import LoopIteration, ReformulationOutput, ReformulationStrategy

class MyReformulator:
    def reformulate(
        self, original_query: str, history: list[LoopIteration]
    ) -> ReformulationOutput:
        last = history[-1]
        reason = last.evaluation.reason

        # Use the history to avoid repeating the same strategy
        used_queries = {it.query for it in history}

        # ... strategy selection logic ...

        return ReformulationOutput(
            queries=["new reformulated query"],
            strategy=ReformulationStrategy.EXPAND,
            reasoning="Added synonyms because reason mentioned vocabulary mismatch.",
        )
```

**Key points:**
- Return at least one query (`queries` has `min_length=1`).
- You receive the ORIGINAL user query, not the reformulated one from the last iteration. Use `history[-1].query` if you need to see what retrieval query was last used.
- The full `history` lets you avoid repeating strategies that failed. If EXPAND didn't work on iteration 1, try NARROW on iteration 2.
- For DECOMPOSE: return multiple queries. ragsnag handles merging, deduplication, and sorting automatically.
- `strategy` and `reasoning` are stored in the trace. Write a descriptive `reasoning` string — it helps you understand what the reformulator decided when debugging.

---

## Combining components

You can mix and match built-in and custom components freely:

```python
loop = RAGLoop(
    retriever=MyPineconeRetriever(),     # custom
    generator=MyClaudeGenerator(),       # custom
    evaluator=LLMEvaluator(my_llm_fn),  # built-in
    reformulator=HeuristicReformulator(),# built-in
)
```

Or use all custom:

```python
loop = RAGLoop(
    retriever=MyRetriever(),
    generator=MyGenerator(),
    evaluator=MyEvaluator(),
    reformulator=MyReformulator(),
)
```

---

## Verifying protocol conformance

If you want to confirm your implementation satisfies a protocol at runtime:

```python
from ragsnag import Retriever, Generator, Evaluator, Reformulator

assert isinstance(MyRetriever(), Retriever)
assert isinstance(MyGenerator(), Generator)
assert isinstance(MyEvaluator(), Evaluator)
assert isinstance(MyReformulator(), Reformulator)
```

This checks that the required method names exist on the class. It does not check return types or parameter types — those are enforced by your type checker (mypy/pyright).
