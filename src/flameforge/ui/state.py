"""Mutable session state shared across TUI screens.

Each screen reads from and writes to a single :class:`SessionState` instance held
by the app. This keeps the linear pipeline (model → method → data → config →
train → export) decoupled: a screen only needs the app's ``session``, not a
reference to its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from flameforge.config import DataConfig, TrainingConfig
from flameforge.constants import ExportFormat, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.models.registry import ModelInfo


@dataclass
class SessionState:
    """Holds every choice the user makes as they move through the pipeline.

    Fields are populated step by step and read by later screens. ``None`` means
    "not chosen yet". The training/export screens require the earlier fields to
    be set; navigation guards enforce that ordering.

    Attributes:
        device: The detected compute device (set at startup).
        model_id: Chosen HuggingFace id or local path.
        model_info: Registry metadata if the model is a known one.
        method: Chosen fine-tuning method.
        data: The configured dataset.
        training: The (possibly auto-tuned) training hyperparameters.
        chat_template: Resolved chat-template key for formatting.
        export_formats: Export artifacts the user requested.
        output_dir: Where artifacts are written.
    """

    device: DeviceInfo
    model_id: str | None = None
    model_info: ModelInfo | None = None
    method: FineTuningMethod | None = None
    data: DataConfig | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    chat_template: str | None = None
    export_formats: list[ExportFormat] = field(default_factory=lambda: [ExportFormat.ADAPTER])
    output_dir: Path = field(default_factory=lambda: Path("./flameforge-output"))

    def reset_downstream_of_model(self) -> None:
        """Clear choices that depend on the model when the model changes."""
        self.method = None
        self.chat_template = None

    @property
    def model_param_count(self) -> int | None:
        """Parameter count of the selected model, if known from the registry."""
        return self.model_info.param_count if self.model_info else None
