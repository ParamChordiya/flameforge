"""Pydantic configuration models for a FlameForge fine-tuning run.

These models are the typed contract shared across the whole codebase. The TUI
populates them step by step; the training backends consume them. All values are
validated at assignment so an invalid configuration can never silently reach the
trainer.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flameforge.constants import (
    DEFAULT_HYPERPARAMS,
    DataFormat,
    ExportFormat,
    FineTuningMethod,
)
from flameforge.utils.errors import ConfigurationError


class TrainingConfig(BaseModel):
    """All tunable hyperparameters for a single training run.

    Defaults mirror :data:`flameforge.constants.DEFAULT_HYPERPARAMS` and
    ``configs/default.yaml``. Field constraints reject obviously invalid values
    (e.g. a non-positive learning rate) at construction time.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Core training schedule.
    num_epochs: int = Field(default=3, ge=1, le=100, description="Number of passes over the dataset.")
    learning_rate: float = Field(default=2.0e-4, gt=0, le=1.0, description="Peak learning rate.")
    lr_scheduler: str = Field(default="cosine", description="LR scheduler type.")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5, description="Fraction of steps spent warming up.")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0, description="AdamW weight decay.")
    max_seq_length: int = Field(default=2048, ge=8, le=131072, description="Maximum tokens per example.")

    # LoRA / DoRA adapter configuration.
    lora_rank: int = Field(default=16, ge=1, le=512, description="LoRA rank (r).")
    lora_alpha: int = Field(default=32, ge=1, le=1024, description="LoRA alpha scaling factor.")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.9, description="LoRA dropout probability.")
    lora_target_modules: str = Field(default="auto", description="Target modules, or 'auto' to let PEFT decide.")

    # Data handling.
    train_eval_split: float = Field(default=0.95, gt=0.0, lt=1.0, description="Fraction of data used for training.")
    shuffle: bool = Field(default=True, description="Whether to shuffle the dataset before splitting.")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility.")

    # Optimization / precision.
    gradient_checkpointing: bool = Field(default=True, description="Trade compute for memory.")
    bf16: bool = Field(default=True, description="Use bfloat16; auto-downgraded to fp16 if unsupported.")
    fp16: bool = Field(default=False, description="Use float16 mixed precision.")
    per_device_batch_size: int = Field(default=1, ge=1, le=256, description="Micro-batch size per device.")
    gradient_accumulation: int = Field(default=1, ge=1, le=512, description="Gradient accumulation steps.")
    effective_batch_size: int = Field(default=32, ge=1, le=4096, description="Target effective batch size.")

    # Checkpointing.
    save_steps: int = Field(default=500, ge=1, description="Save a checkpoint every N steps.")
    save_total_limit: int = Field(default=3, ge=1, description="Maximum number of checkpoints to retain.")
    save_best: bool = Field(default=True, description="Keep the checkpoint with the lowest eval loss.")

    output_dir: str = Field(default=str(DEFAULT_HYPERPARAMS["output_dir"]), description="Where to write artifacts.")

    @field_validator("lr_scheduler")
    @classmethod
    def _validate_scheduler(cls, value: str) -> str:
        allowed = {"linear", "cosine", "constant", "constant_with_warmup", "polynomial"}
        if value not in allowed:
            raise ValueError(f"lr_scheduler must be one of {sorted(allowed)}, got '{value}'.")
        return value

    @model_validator(mode="after")
    def _validate_precision(self) -> TrainingConfig:
        if self.bf16 and self.fp16:
            raise ValueError("Only one of bf16/fp16 may be enabled at a time.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load a :class:`TrainingConfig` from a YAML file.

        Args:
            path: Path to a YAML file of hyperparameter overrides.

        Returns:
            A validated :class:`TrainingConfig`.

        Raises:
            ConfigurationError: If the file is missing or fails validation.
        """
        p = Path(path)
        if not p.is_file():
            raise ConfigurationError(
                message=f"Config file not found: {p}",
                suggestions=["Check the path, or omit --config to use built-in defaults."],
            )
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - exercised via error path
            raise ConfigurationError(
                message=f"Could not parse YAML config: {p}",
                details=str(exc),
            ) from exc
        try:
            return cls(**data)
        except Exception as exc:  # pydantic ValidationError → friendly message
            raise ConfigurationError(
                message=f"Invalid values in config file: {p}",
                details=str(exc),
            ) from exc

    def effective_batch(self) -> int:
        """Return the realized effective batch size.

        Returns:
            ``per_device_batch_size * gradient_accumulation``.
        """
        return self.per_device_batch_size * self.gradient_accumulation


class DataConfig(BaseModel):
    """Describes the dataset the user has chosen and how to interpret it."""

    model_config = ConfigDict(validate_assignment=True)

    path: Path
    data_format: DataFormat
    # For CSV/TSV: maps a logical field ("instruction"/"output"/...) to a column.
    column_mapping: dict[str, str] = Field(default_factory=dict)
    num_examples: int = Field(default=0, ge=0)

    @field_validator("path")
    @classmethod
    def _path_exists(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Data file does not exist: {value}")
        return value


class RunConfig(BaseModel):
    """Top-level configuration tying together every choice for a run.

    This is the object handed to a trainer. It is assembled incrementally by the
    TUI as the user progresses through the pipeline.
    """

    model_config = ConfigDict(validate_assignment=True)

    model_id: str = Field(description="HuggingFace model id or local path.")
    method: FineTuningMethod
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    chat_template: str | None = Field(default=None, description="Chat-template key override.")
    export_formats: list[ExportFormat] = Field(default_factory=lambda: [ExportFormat.ADAPTER])

    @property
    def uses_adapter(self) -> bool:
        """Whether this run produces a LoRA-style adapter (vs. full fine-tuning)."""
        return self.method != FineTuningMethod.FULL
