"""Memory budgeting, model-size estimation, and runtime pressure monitoring.

Memory safety is critical on Apple Silicon, where GPU work shares unified memory
with macOS. Over-committing can freeze the whole machine, so FlameForge always
budgets conservatively and monitors usage during training.

To avoid a hard dependency on ``torch``/``mlx`` (and on the ``DeviceInfo`` type,
which would create an import cycle), the functions here accept a small structural
protocol rather than the concrete dataclass.
"""

from __future__ import annotations

from typing import Protocol

import psutil

from flameforge.constants import (
    BYTES_PER_PARAM,
    MEMORY_BUDGET_FRACTION,
    DeviceType,
    FineTuningMethod,
)


class _DeviceLike(Protocol):
    """Structural type for anything carrying a device type and total memory.

    Members are declared read-only so a frozen dataclass (``DeviceInfo``)
    structurally satisfies the protocol.
    """

    @property
    def device_type(self) -> DeviceType: ...

    @property
    def total_memory_gb(self) -> float: ...


def safe_budget_fraction(device: _DeviceLike) -> float:
    """Return the conservative budget fraction for a device type.

    CUDA VRAM is dedicated (90%); Apple Silicon unified memory must leave room
    for macOS (70%); CPU RAM is capped at 50%.

    Args:
        device: An object exposing ``device_type``.

    Returns:
        The safe fraction of total memory to use for training.
    """
    return MEMORY_BUDGET_FRACTION.get(device.device_type, 0.5)


def default_memory_budget(device: _DeviceLike) -> float:
    """Return the conservative default budget (the safe fraction of total)."""
    return round(device.total_memory_gb * safe_budget_fraction(device), 2)


def calculate_memory_budget(device: _DeviceLike, user_cap_gb: float | None = None) -> float:
    """Compute the training memory budget for a device.

    With no user value, returns the conservative default (the safe fraction of
    total memory). When the user explicitly chooses a budget, that value is used
    directly — it may lower the budget for extra safety, or raise it above the
    default (riskier; the UI/CLI warns about this). The budget never exceeds the
    device's total physical memory.

    Args:
        device: An object exposing ``device_type`` and ``total_memory_gb``.
        user_cap_gb: Optional user-chosen budget, in GB.

    Returns:
        The training memory budget in gigabytes.
    """
    if user_cap_gb is None:
        return default_memory_budget(device)
    budget = min(user_cap_gb, device.total_memory_gb)
    return round(max(0.0, budget), 2)


def exceeds_safe_budget(device: _DeviceLike, budget_gb: float) -> bool:
    """Return whether ``budget_gb`` is above the conservative safe default.

    Args:
        device: The device the budget is for.
        budget_gb: The proposed budget.

    Returns:
        True if the budget exceeds the safe fraction of total memory (risky).
    """
    return budget_gb > default_memory_budget(device) + 1e-6


def estimate_model_memory_gb(param_count: int, method: FineTuningMethod | str, dtype: str = "bfloat16") -> float:
    """Estimate peak training memory for a model under a given method.

    The estimate accounts for the base weights plus method-specific overhead:
    QLoRA loads the base in 4-bit, LoRA/DoRA keep it in 16-bit with small adapter
    and optimizer state, and full fine-tuning needs roughly 4x the weights for
    gradients and AdamW optimizer states.

    Args:
        param_count: Number of model parameters.
        method: The fine-tuning method (enum or its string value).
        dtype: The base-weight dtype for non-quantized methods.

    Returns:
        An estimate of peak training memory in gigabytes.

    Raises:
        KeyError: If ``dtype`` is not a recognized precision.
    """
    method_value = method.value if isinstance(method, FineTuningMethod) else str(method)
    bytes_per_param = BYTES_PER_PARAM[dtype]
    base = param_count * bytes_per_param / 1e9

    if method_value == FineTuningMethod.QLORA.value:
        # 4-bit base weights + adapter/optimizer overhead.
        return round((param_count * 0.5 / 1e9) + 1.5, 2)
    if method_value in (FineTuningMethod.LORA.value, FineTuningMethod.DORA.value):
        # 16-bit base + adapters + optimizer state for the small trainable set.
        return round(base + 2.0, 2)
    if method_value == FineTuningMethod.FULL.value:
        # Weights + gradients + AdamW (2x): ~4x the base weights.
        return round(base * 4, 2)
    return round(base, 2)


def fits_in_budget(param_count: int, method: FineTuningMethod | str, budget_gb: float, dtype: str = "bfloat16") -> bool:
    """Return whether a model/method is expected to fit within a memory budget.

    Args:
        param_count: Number of model parameters.
        method: The fine-tuning method.
        budget_gb: The available memory budget in GB.
        dtype: Base-weight dtype for non-quantized methods.

    Returns:
        True if the estimated memory is at or below the budget.
    """
    return estimate_model_memory_gb(param_count, method, dtype) <= budget_gb


def current_process_memory_gb() -> float:
    """Return the current process's resident memory in gigabytes.

    Returns:
        Resident set size (RSS) in GB. Used as a portable fallback when backend
        specific memory counters are unavailable.
    """
    return float(psutil.Process().memory_info().rss) / 1e9


def system_memory_used_fraction() -> float:
    """Return the fraction of total system memory currently in use.

    Returns:
        A value in [0, 1]; on Apple Silicon this is a good proxy for unified
        memory pressure.
    """
    return float(psutil.virtual_memory().percent) / 100.0
