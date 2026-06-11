"""Tests for the model registry and its memory helpers."""

from __future__ import annotations

import pytest

from flameforge.constants import FineTuningMethod
from flameforge.models.registry import (
    POPULAR_MODELS,
    ModelInfo,
    get_model,
    models_fitting_budget,
    search_models,
)


def test_registry_is_populated() -> None:
    assert len(POPULAR_MODELS) >= 15


def test_registry_ids_are_unique() -> None:
    ids = [m.hf_id for m in POPULAR_MODELS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("model", POPULAR_MODELS, ids=lambda m: m.hf_id)
def test_each_model_is_well_formed(model: ModelInfo) -> None:
    assert model.param_count > 0
    assert model.context_length > 0
    assert model.recommended_methods, "must recommend at least one method"
    # Every recommended method must have a memory estimate.
    for method in model.recommended_methods:
        assert method in model.min_memory_gb
    # Cheapest method is among the recommended ones.
    assert model.cheapest_method() in model.recommended_methods


def test_get_model_hit_and_miss() -> None:
    assert get_model("mistralai/Mistral-7B-Instruct-v0.3") is not None
    assert get_model("nonexistent/model") is None


def test_search_models_by_family_and_name() -> None:
    qwen = search_models("qwen")
    assert qwen and all(m.family == "qwen" for m in qwen)
    assert search_models("") == POPULAR_MODELS
    assert search_models("Mistral 7B")


def test_param_count_str() -> None:
    m = get_model("meta-llama/Llama-3.1-8B-Instruct")
    assert m is not None and m.param_count_str == "8.0B"
    phi = get_model("microsoft/Phi-3-mini-4k-instruct")
    assert phi is not None and phi.param_count_str == "3.8B"
    tiny = get_model("Qwen/Qwen2.5-0.5B-Instruct")
    assert tiny is not None and tiny.param_count_str == "500M"


def test_models_fitting_budget_monotonic() -> None:
    small = models_fitting_budget(4.0)
    large = models_fitting_budget(50.0)
    assert set(m.hf_id for m in small).issubset(m.hf_id for m in large)
    # A tiny model fits a small budget; the 70B does not.
    assert any(m.param_count <= 1_500_000_000 for m in small)
    assert all(m.param_count < 70_000_000_000 for m in small)


def test_estimated_memory_matches_helper() -> None:
    m = get_model("Qwen/Qwen2.5-7B-Instruct")
    assert m is not None
    assert m.estimated_memory_gb(FineTuningMethod.QLORA) < m.estimated_memory_gb(FineTuningMethod.LORA)
