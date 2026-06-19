# Changelog

All notable changes to ragsnag will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] - 2026-06-20

### Added
- `RAGLoop` — core loop engine with configurable max iterations and confidence threshold
- `LLMEvaluator` — evaluates answer grounding and completeness using any LLM
- `LLMReformulator` — rewrites queries using automatic strategy selection (expand, narrow, decompose, step_back, hyde, perspective_shift)
- `HeuristicReformulator` — zero-cost reformulation using keyword-based strategy selection
- Four pluggable protocols: `Retriever`, `Generator`, `Evaluator`, `Reformulator`
- `LoopResult` with full iteration trace and `best_iteration` property
- `LoopConfig` with `on_iteration` callback hook
- Multi-query support for `decompose` strategy — chunks merged and deduplicated across sub-queries
- 119 tests across all components
- GitHub Actions CI on Python 3.10, 3.11, 3.12
- Examples for Claude and OpenAI integration
