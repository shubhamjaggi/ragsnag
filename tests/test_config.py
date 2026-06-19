"""Tests for LoopConfig validation."""
from __future__ import annotations

import pytest

from ragsnag._config import LoopConfig


def test_default_values() -> None:
    config = LoopConfig()
    assert config.max_iterations == 3
    assert config.confidence_threshold == 0.8
    assert config.top_k == 5
    assert config.on_iteration is None


def test_valid_custom_config() -> None:
    config = LoopConfig(max_iterations=5, confidence_threshold=0.9, top_k=10)
    assert config.max_iterations == 5
    assert config.confidence_threshold == 0.9
    assert config.top_k == 10


def test_max_iterations_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        LoopConfig(max_iterations=0)


def test_max_iterations_negative_raises() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        LoopConfig(max_iterations=-1)


def test_max_iterations_one_is_valid() -> None:
    config = LoopConfig(max_iterations=1)
    assert config.max_iterations == 1


def test_confidence_threshold_above_one_raises() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        LoopConfig(confidence_threshold=1.1)


def test_confidence_threshold_negative_raises() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        LoopConfig(confidence_threshold=-0.1)


def test_confidence_threshold_boundary_zero() -> None:
    config = LoopConfig(confidence_threshold=0.0)
    assert config.confidence_threshold == 0.0


def test_confidence_threshold_boundary_one() -> None:
    config = LoopConfig(confidence_threshold=1.0)
    assert config.confidence_threshold == 1.0


def test_top_k_zero_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        LoopConfig(top_k=0)


def test_top_k_negative_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        LoopConfig(top_k=-5)


def test_top_k_one_is_valid() -> None:
    config = LoopConfig(top_k=1)
    assert config.top_k == 1


def test_on_iteration_callback_stored() -> None:
    cb = lambda it: None  # noqa: E731
    config = LoopConfig(on_iteration=cb)
    assert config.on_iteration is cb
