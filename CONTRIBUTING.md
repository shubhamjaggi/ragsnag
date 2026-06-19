# Contributing to ragsnag

Thanks for taking the time to contribute.

## Development setup

```bash
git clone https://github.com/shubhamjaggi/ragsnag
cd ragsnag
pip install -e ".[dev]"
```

Install pre-commit hooks (optional but recommended):

```bash
pip install pre-commit
pre-commit install
```

## Running tests

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=ragsnag --cov-report=term-missing
```

## Lint and format

```bash
ruff check src/        # lint
ruff format src/       # format
```

## Type checking

```bash
mypy src/ragsnag
```

## Project structure

```
src/ragsnag/
  _models.py        — Pydantic data models (Chunk, LoopResult, etc.)
  _protocols.py     — Pluggable interfaces (Retriever, Generator, etc.)
  _config.py        — LoopConfig
  _loop.py          — Core loop engine
  _evaluator.py     — LLMEvaluator
  _reformulator.py  — LLMReformulator + HeuristicReformulator

tests/
  conftest.py       — Shared mocks and helpers
  test_loop.py
  test_evaluator.py
  test_reformulator.py
  test_models.py
  test_config.py
  test_protocols.py

examples/           — Reference integrations (not packaged)
```

## Guidelines

**Scope** — ragsnag owns only the loop. It has no opinion on which LLM or vector DB you use. PRs that add vendor-specific code to the core library will not be merged. Vendor integrations belong in `examples/`.

**Tests** — every change needs tests. Coverage must stay above 90%. New behavior without tests will not be merged.

**No bloat** — avoid adding abstractions, fallbacks, or features that no one has asked for. If in doubt, open an issue first.

**One thing per PR** — keep PRs focused. A bug fix and a new feature should be separate PRs.

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Ensure all checks pass: `pytest`, `ruff check src/`, `mypy src/ragsnag`
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Open a PR against `main`

## Reporting bugs

Open a [bug report](https://github.com/shubhamjaggi/ragsnag/issues/new?template=bug_report.md) with a minimal reproduction.

## Suggesting features

Open a [feature request](https://github.com/shubhamjaggi/ragsnag/issues/new?template=feature_request.md) describing the problem you want solved.
