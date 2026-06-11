"""Tests for device detection and memory budgeting."""

from __future__ import annotations

import pytest

from flameforge.constants import Backend, DeviceType, FineTuningMethod
from flameforge.device.detector import DeviceInfo, detect_device, with_memory_cap
from flameforge.device.memory import (
    calculate_memory_budget,
    estimate_model_memory_gb,
    exceeds_safe_budget,
    fits_in_budget,
)


def test_detect_device_returns_valid_info() -> None:
    info = detect_device()
    assert isinstance(info, DeviceInfo)
    assert info.device_type in DeviceType
    assert info.backend in Backend
    assert info.total_memory_gb > 0
    assert 0 < info.memory_budget_gb <= info.total_memory_gb


def test_detect_device_respects_user_cap() -> None:
    info = detect_device(user_cap_gb=1.0)
    assert info.memory_budget_gb <= 1.0


def test_budget_fractions(cuda_device: DeviceInfo, mps_device: DeviceInfo, cpu_device: DeviceInfo) -> None:
    assert calculate_memory_budget(cuda_device) == pytest.approx(24.0 * 0.90)
    assert calculate_memory_budget(mps_device) == pytest.approx(16.0 * 0.70)
    assert calculate_memory_budget(cpu_device) == pytest.approx(8.0 * 0.50)


def test_budget_user_value_lowers_and_raises(cuda_device: DeviceInfo) -> None:
    # A value below the default lowers the budget (safer).
    assert calculate_memory_budget(cuda_device, user_cap_gb=5.0) == 5.0
    # A value above the default raises it (riskier), but never beyond physical RAM.
    assert calculate_memory_budget(cuda_device, user_cap_gb=23.0) == 23.0
    assert calculate_memory_budget(cuda_device, user_cap_gb=100.0) == pytest.approx(24.0)


def test_exceeds_safe_budget(mps_device: DeviceInfo) -> None:
    # 16 GB Mac → safe default is 11.2 GB.
    assert exceeds_safe_budget(mps_device, 14.0) is True
    assert exceeds_safe_budget(mps_device, 10.0) is False


def test_with_memory_cap_updates_budget(mps_device: DeviceInfo) -> None:
    capped = with_memory_cap(mps_device, 8.0)
    assert capped.memory_budget_gb == 8.0
    assert capped.total_memory_gb == mps_device.total_memory_gb
    # Resetting with None returns the conservative default (70% of 16).
    reset = with_memory_cap(capped, None)
    assert reset.memory_budget_gb == pytest.approx(11.2)


def test_estimate_qlora_is_cheapest() -> None:
    params = 7_000_000_000
    qlora = estimate_model_memory_gb(params, FineTuningMethod.QLORA)
    lora = estimate_model_memory_gb(params, FineTuningMethod.LORA)
    full = estimate_model_memory_gb(params, FineTuningMethod.FULL)
    assert qlora < lora < full


def test_estimate_accepts_string_method() -> None:
    assert estimate_model_memory_gb(1_000_000_000, "qlora") == estimate_model_memory_gb(
        1_000_000_000, FineTuningMethod.QLORA
    )


def test_full_finetune_is_four_times_base() -> None:
    params = 1_000_000_000
    base_bf16 = params * 2 / 1e9
    assert estimate_model_memory_gb(params, FineTuningMethod.FULL) == pytest.approx(base_bf16 * 4)


def test_fits_in_budget() -> None:
    assert fits_in_budget(1_000_000_000, FineTuningMethod.QLORA, budget_gb=4.0)
    assert not fits_in_budget(70_000_000_000, FineTuningMethod.FULL, budget_gb=24.0)


def test_summary_lines_mention_budget(mps_device: DeviceInfo) -> None:
    text = "\n".join(mps_device.summary_lines())
    assert "Budget" in text
    assert "Apple M2 Pro" in text


def test_cpu_only_flag(cpu_device: DeviceInfo, cuda_device: DeviceInfo) -> None:
    assert cpu_device.is_cpu_only is True
    assert cuda_device.is_cpu_only is False
