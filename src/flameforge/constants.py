"""Project-wide constants: enums, default hyperparameters, and memory factors.

This module holds pure constant data with no dependencies on other FlameForge
modules, so it can be imported anywhere without risk of circular imports. Higher
level modules (``config``, ``device``, ``models``) build on these values.
"""

from __future__ import annotations

from enum import Enum

APP_NAME = "FlameForge"
LOG_FILENAME = "flameforge.log"
DEFAULT_OUTPUT_DIR = "./flameforge-output"


class DeviceType(str, Enum):
    """The kind of compute device FlameForge is running on."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class Backend(str, Enum):
    """The training backend used for a given device."""

    PYTORCH = "pytorch"
    MLX = "mlx"


class FineTuningMethod(str, Enum):
    """Supported parameter-efficient and full fine-tuning methods."""

    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"
    DORA = "dora"


class DataFormat(str, Enum):
    """Recognised dataset formats."""

    ALPACA = "alpaca"
    CONVERSATIONAL = "conversational"
    TEXT = "text"
    CSV = "csv"


class ExportFormat(str, Enum):
    """Supported export artifact types."""

    ADAPTER = "adapter"
    MERGED = "merged"
    GGUF = "gguf"


# Bytes consumed per parameter for each dtype, used in memory estimation.
BYTES_PER_PARAM: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Fraction of total memory considered a safe training budget, per device type.
# Apple Silicon uses unified memory shared with the OS, so we are conservative.
MEMORY_BUDGET_FRACTION: dict[DeviceType, float] = {
    DeviceType.CUDA: 0.90,
    DeviceType.MPS: 0.70,
    DeviceType.CPU: 0.50,
}

# Runtime memory-pressure thresholds (as a fraction of the budget) that trigger
# automatic mitigation during training.
MEMORY_REDUCE_BATCH_THRESHOLD = 0.90
MEMORY_PAUSE_THRESHOLD = 0.95

# Default training hyperparameters. Mirrors ``configs/default.yaml`` and is the
# single source of truth consumed by :class:`flameforge.config.TrainingConfig`.
DEFAULT_HYPERPARAMS: dict[str, object] = {
    "num_epochs": 3,
    "learning_rate": 2.0e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "auto",
    "train_eval_split": 0.95,
    "shuffle": True,
    "seed": 42,
    "gradient_checkpointing": True,
    "bf16": True,
    "effective_batch_size": 32,
    "save_steps": 500,
    "save_total_limit": 3,
    "save_best": True,
    "output_dir": DEFAULT_OUTPUT_DIR,
}

# GGUF quantization levels offered at export time, best-quality first.
GGUF_QUANT_LEVELS = ["Q8_0", "Q5_K_M", "Q4_K_M", "F16"]
