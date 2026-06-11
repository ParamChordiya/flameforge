"""Auto-tune training hyperparameters from model, device, and dataset context.

The goal is sensible defaults a beginner can accept blindly: fewer epochs and a
gentler learning rate for tiny datasets (to avoid overfitting), a larger LoRA
rank for big datasets, a bf16→fp16 downgrade on older GPUs, and a memory-aware
micro-batch-size guess. Every adjustment is returned alongside a plain-language
warning so the config screen can explain *why* a value was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

from flameforge.config import TrainingConfig
from flameforge.constants import DeviceType, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.device.memory import estimate_model_memory_gb
from flameforge.models.registry import ModelInfo


@dataclass
class AutoTuneResult:
    """The outcome of auto-tuning.

    Attributes:
        config: The adjusted training configuration.
        warnings: Plain-language notes about the adjustments made.
    """

    config: TrainingConfig
    warnings: list[str]


def suggest_batch_size(
    param_count: int | None,
    method: FineTuningMethod,
    max_seq_length: int,
    budget_gb: float,
) -> int:
    """Heuristically pick a per-device micro-batch size that should fit memory.

    A real probe (forward/backward at increasing sizes) happens inside the CUDA
    trainer; this is the pre-flight estimate shown in the UI. It scales the
    headroom left after the model weights by an approximate per-sample cost.

    Args:
        param_count: Model parameter count, or None if unknown.
        method: The fine-tuning method.
        max_seq_length: Sequence length per example.
        budget_gb: The memory budget in GB.

    Returns:
        A micro-batch size between 1 and 32.
    """
    if param_count is None:
        return 1
    model_gb = estimate_model_memory_gb(param_count, method)
    headroom = max(0.0, budget_gb - model_gb)
    # Approximate activation memory per sample (GB), scaling with sequence length
    # and model size; deliberately conservative.
    per_sample = max(0.05, (param_count / 1e9) * (max_seq_length / 1024) * 0.12)
    raw = int(headroom / per_sample) if per_sample > 0 else 1
    # Snap down to a power of two for kernel friendliness, clamp to [1, 32].
    size = 1
    while size * 2 <= raw and size < 32:
        size *= 2
    return max(1, size)


def auto_tune(
    config: TrainingConfig,
    method: FineTuningMethod,
    device: DeviceInfo,
    dataset_size: int,
    model_info: ModelInfo | None = None,
) -> AutoTuneResult:
    """Return a context-adjusted copy of ``config`` plus explanatory warnings.

    Args:
        config: The starting configuration (typically the defaults).
        method: The chosen fine-tuning method.
        device: The detected device.
        dataset_size: Number of training examples.
        model_info: Registry metadata for the model, if known.

    Returns:
        An :class:`AutoTuneResult` with the adjusted config and warnings.
    """
    tuned = config.model_copy(deep=True)
    warnings: list[str] = []

    # -- Dataset-size driven adjustments --------------------------------
    if dataset_size < 100:
        tuned.num_epochs = min(tuned.num_epochs, 1)
        tuned.learning_rate = round(tuned.learning_rate * 0.5, 8)
        warnings.append(
            f"Small dataset ({dataset_size} examples): reduced to 1 epoch and halved the "
            "learning rate to prevent overfitting."
        )
    elif dataset_size < 500:
        if tuned.num_epochs > 2:
            tuned.num_epochs = 2
            warnings.append(f"Modest dataset ({dataset_size} examples): capped epochs at 2.")

    if dataset_size > 50_000:
        tuned.learning_rate = min(tuned.learning_rate, 1.0e-4)
        warnings.append("Large dataset (>50k): lowered the learning rate to 1e-4 for stability.")

    if dataset_size > 10_000:
        tuned.lora_rank = max(tuned.lora_rank, 32)
        tuned.lora_alpha = max(tuned.lora_alpha, 64)
        warnings.append("Large dataset (>10k): raised LoRA rank to 32 (alpha 64) for more capacity.")

    # -- Precision adjustments ------------------------------------------
    if (
        device.device_type == DeviceType.CUDA
        and device.compute_capability is not None
        and device.compute_capability < "8.0"
    ):
        tuned.bf16 = False
        tuned.fp16 = True
        warnings.append(f"GPU compute capability {device.compute_capability} lacks bf16: switched to fp16.")

    # -- Batch size / accumulation --------------------------------------
    param_count = model_info.param_count if model_info else None
    micro = suggest_batch_size(param_count, method, tuned.max_seq_length, device.memory_budget_gb)
    tuned.per_device_batch_size = micro
    tuned.gradient_accumulation = max(1, tuned.effective_batch_size // micro)
    if param_count is not None:
        warnings.append(
            f"Set micro-batch {micro} × {tuned.gradient_accumulation} accumulation "
            f"≈ effective batch {micro * tuned.gradient_accumulation}."
        )

    return AutoTuneResult(config=tuned, warnings=warnings)
