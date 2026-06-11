"""Tests for hyperparameter auto-tuning and batch-size heuristics."""

from __future__ import annotations

import pytest

from flameforge.config import TrainingConfig
from flameforge.constants import Backend, DeviceType, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.models.registry import get_model
from flameforge.training.hyperparams import auto_tune, suggest_batch_size


def _cuda(capability: str = "8.9") -> DeviceInfo:
    return DeviceInfo(DeviceType.CUDA, "RTX", 24.0, 21.6, capability, 1, Backend.PYTORCH)


def _mps() -> DeviceInfo:
    return DeviceInfo(DeviceType.MPS, "M", 18.0, 12.6, None, 1, Backend.MLX)


def test_small_dataset_reduces_epochs_and_lr() -> None:
    base = TrainingConfig()
    result = auto_tune(base, FineTuningMethod.LORA, _mps(), dataset_size=50)
    assert result.config.num_epochs == 1
    assert result.config.learning_rate < base.learning_rate
    assert any("Small dataset" in w for w in result.warnings)


def test_modest_dataset_caps_epochs() -> None:
    result = auto_tune(TrainingConfig(num_epochs=3), FineTuningMethod.LORA, _mps(), dataset_size=300)
    assert result.config.num_epochs == 2


def test_large_dataset_lowers_lr_and_raises_rank() -> None:
    result = auto_tune(TrainingConfig(), FineTuningMethod.LORA, _cuda(), dataset_size=60_000)
    assert result.config.learning_rate <= 1.0e-4
    assert result.config.lora_rank >= 32
    assert result.config.lora_alpha >= 64


def test_old_gpu_downgrades_to_fp16() -> None:
    result = auto_tune(TrainingConfig(), FineTuningMethod.LORA, _cuda(capability="7.5"), dataset_size=1000)
    assert result.config.bf16 is False
    assert result.config.fp16 is True


def test_modern_gpu_keeps_bf16() -> None:
    result = auto_tune(TrainingConfig(), FineTuningMethod.LORA, _cuda(capability="8.9"), dataset_size=1000)
    assert result.config.bf16 is True


def test_auto_tune_does_not_mutate_input() -> None:
    base = TrainingConfig(num_epochs=3)
    auto_tune(base, FineTuningMethod.LORA, _mps(), dataset_size=10)
    assert base.num_epochs == 3  # original untouched


def test_suggest_batch_size_bounds() -> None:
    assert suggest_batch_size(None, FineTuningMethod.LORA, 2048, 12) == 1
    big = suggest_batch_size(70_000_000_000, FineTuningMethod.QLORA, 2048, 24)
    small_model = suggest_batch_size(500_000_000, FineTuningMethod.LORA, 512, 16)
    assert big == 1
    assert 1 <= small_model <= 32
    # Power of two.
    assert small_model & (small_model - 1) == 0


def test_auto_tune_sets_accumulation() -> None:
    info = get_model("Qwen/Qwen2.5-3B-Instruct")
    result = auto_tune(TrainingConfig(effective_batch_size=32), FineTuningMethod.LORA, _mps(), 2000, info)
    cfg = result.config
    assert cfg.per_device_batch_size >= 1
    assert cfg.gradient_accumulation >= 1
    assert cfg.per_device_batch_size * cfg.gradient_accumulation == pytest.approx(32, abs=cfg.per_device_batch_size)
