"""CUDA training backend built on HuggingFace TRL's ``SFTTrainer``.

Construction is cheap and only checks that the required libraries are importable
(raising :class:`~flameforge.utils.errors.DependencyMissingError` otherwise so the
caller can fall back to the simulator). The heavy lifting — loading the model,
attaching PEFT adapters, and running ``SFTTrainer`` — happens in :meth:`train`,
which drives the underlying loop on a worker thread and streams metrics back.
"""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Iterator
from pathlib import Path

from flameforge.config import RunConfig
from flameforge.constants import FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.training.base import BaseTrainer, TrainingMetrics, TrainingResult
from flameforge.training.callbacks import MetricStream, build_hf_callback
from flameforge.utils.errors import DependencyMissingError, TrainingError
from flameforge.utils.logging import get_logger

_log = get_logger("training.cuda_trainer")

_REQUIRED = ("torch", "transformers", "trl", "peft", "datasets")


class CudaTrainer(BaseTrainer):
    """Trains a model on CUDA using TRL + PEFT.

    Raises:
        DependencyMissingError: At construction, if any required library
            (``torch``/``transformers``/``trl``/``peft``/``datasets``) is missing.
    """

    def __init__(self, run_config: RunConfig, device: DeviceInfo, num_examples: int) -> None:
        missing = [pkg for pkg in _REQUIRED if importlib.util.find_spec(pkg) is None]
        if missing:
            raise DependencyMissingError(
                package=" ".join(missing),
                reason="CUDA training",
                extra_notes=["Install the CUDA extra: pip install 'flameforge[cuda]'."],
            )
        super().__init__(run_config, device, num_examples)
        self._trainer: object | None = None

    def _peft_config(self) -> object | None:
        """Build a PEFT LoRA/DoRA config, or None for full fine-tuning."""
        method = self.run_config.method
        if method == FineTuningMethod.FULL:
            return None
        from peft import LoraConfig, TaskType  # type: ignore[import-not-found]

        cfg = self.run_config.training
        config: object = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_dora=method == FineTuningMethod.DORA,
        )
        return config

    def _build_sft_config(self) -> object:
        """Construct a TRL ``SFTConfig`` from the run's training config."""
        from trl import SFTConfig  # type: ignore[import-not-found]

        cfg = self.run_config.training
        return SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            logging_steps=1,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            bf16=cfg.bf16,
            fp16=cfg.fp16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            max_seq_length=cfg.max_seq_length,
            seed=cfg.seed,
            report_to="none",
        )

    def train(self) -> Iterator[TrainingMetrics]:
        """Run SFT training, yielding metrics as they are logged.

        Yields:
            One :class:`TrainingMetrics` per logged step.

        Raises:
            TrainingError: If no data was attached or the loop fails.
        """
        if self.data is None:
            raise TrainingError(
                message="Training data was not prepared before starting.",
                suggestions=["This is an internal error; please re-run the data step."],
            )
        from datasets import Dataset as HFDataset  # type: ignore[import-not-found]

        from flameforge.models.loader import load_model

        loaded = load_model(
            self.run_config.model_id,
            self.run_config.method,
            self.device,
        )
        train_ds = HFDataset.from_dict({"text": self.data.train_texts})
        eval_ds = HFDataset.from_dict({"text": self.data.eval_texts}) if self.data.eval_texts else None

        from trl import SFTTrainer  # type: ignore[import-not-found]

        stream = MetricStream()
        callback = build_hf_callback(stream, self.control, self.device, self.total_steps)
        trainer = SFTTrainer(
            model=loaded.model,
            args=self._build_sft_config(),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=self._peft_config(),
            processing_class=loaded.tokenizer,
            callbacks=[callback],
        )
        self._trainer = trainer

        def run() -> None:
            try:
                trainer.train()
                stream.finish()
            except Exception as exc:  # surface any training failure to the consumer
                _log.exception("CUDA training failed")
                stream.finish(TrainingError(message=f"Training failed: {exc}", details=str(exc)))

        worker = threading.Thread(target=run, name="flameforge-cuda-train", daemon=True)
        worker.start()

        last: TrainingMetrics | None = None
        for metrics in stream:
            last = metrics
            yield metrics
        worker.join(timeout=5)
        self._finalize(last)

    def _finalize(self, last: TrainingMetrics | None) -> None:
        """Record the :class:`TrainingResult` after the loop completes."""
        cfg = self.run_config.training
        self.result = TrainingResult(
            final_loss=last.loss if last else 0.0,
            best_loss=last.best_loss if last and last.best_loss is not None else (last.loss if last else 0.0),
            steps_completed=last.step if last else 0,
            elapsed_sec=last.elapsed_sec if last else 0.0,
            total_tokens=int(last.tokens_per_sec * last.elapsed_sec) if last else 0,
            output_dir=Path(cfg.output_dir),
            stopped_early=self.control.stop_requested.is_set(),
            simulated=False,
        )

    def save_checkpoint(self, step: int) -> Path:
        """Request an immediate checkpoint at the next step boundary.

        Args:
            step: The step label (informational).

        Returns:
            The output directory the checkpoint will be written under.
        """
        self.control.checkpoint_requested.set()
        return Path(self.run_config.training.output_dir)
