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


def calculate_memory_budget(device: _DeviceLike, user_cap_gb: float | None = None) -> float:
    """Compute a safe training memory budget for a device.

    CUDA VRAM is dedicated, so we allow 90%. Apple Silicon unified memory must
    leave room for macOS, so we allow only 70%. CPU RAM is capped at 50%.

    Args:
        device: An object exposing ``device_type`` and ``total_memory_gb``.
        user_cap_gb: Optional user-specified hard cap, in GB. The returned budget
            never exceeds this value.

    Returns:
        The training memory budget in gigabytes.
    """
    fraction = MEMORY_BUDGET_FRACTION.get(device.device_type, 0.5)
    budget = device.total_memory_gb * fraction
    if user_cap_gb is not None:
        budget = min(budget, user_cap_gb)
    return round(budget, 2)


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
    return psutil.Process().memory_info().rss / 1e9


def system_memory_used_fraction() -> float:
    """Return the fraction of total system memory currently in use.

    Returns:
        A value in [0, 1]; on Apple Silicon this is a good proxy for unified
        memory pressure.
    """
    return psutil.virtual_memory().percent / 100.0
