"""
ragsnag — loop engineering for RAG systems.

Iterative retrieval-augmented generation with automatic query reformulation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ragsnag")
except PackageNotFoundError:
    __version__ = "unknown"

from ragsnag._config import LoopConfig
from ragsnag._evaluator import LLMEvaluator
from ragsnag._loop import RAGLoop
from ragsnag._models import (
    Chunk,
    EvaluationResult,
    GeneratorOutput,
    LoopIteration,
    LoopResult,
    ReformulationOutput,
    ReformulationStrategy,
    StopReason,
)
from ragsnag._protocols import Evaluator, Generator, Reformulator, Retriever
from ragsnag._reformulator import HeuristicReformulator, LLMReformulator

__all__ = [
    "__version__",
    # core
    "RAGLoop",
    "LoopConfig",
    # models
    "Chunk",
    "GeneratorOutput",
    "EvaluationResult",
    "LoopIteration",
    "LoopResult",
    "ReformulationOutput",
    "ReformulationStrategy",
    "StopReason",
    # protocols
    "Retriever",
    "Generator",
    "Evaluator",
    "Reformulator",
    # defaults
    "LLMEvaluator",
    "LLMReformulator",
    "HeuristicReformulator",
]
