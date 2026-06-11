"""MLX training backend for Apple Silicon, built on ``mlx-lm``'s tuner.

Like the CUDA backend, construction only checks that ``mlx``/``mlx-lm`` are
importable; the model load, LoRA conversion, and training loop happen in
:meth:`train`. MLX uses unified memory, so this backend reports active GPU memory
each step and periodically clears the cache to keep pressure down.

The training loop runs on a worker thread and streams metrics back through a
:class:`~flameforge.training.callbacks.MetricStream`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from flameforge.config import RunConfig
from flameforge.constants import FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.training.base import BaseTrainer, TrainingMetrics, TrainingResult
from flameforge.training.callbacks import MetricStream
from flameforge.utils.errors import DependencyMissingError, TrainingError
from flameforge.utils.logging import get_logger

_log = get_logger("training.mlx_trainer")

_REQUIRED = ("mlx", "mlx_lm")


class MlxTrainer(BaseTrainer):
    """Trains a model on Apple Silicon using mlx-lm's LoRA tuner.

    Raises:
        DependencyMissingError: At construction, if ``mlx``/``mlx-lm`` are missing.
    """

    def __init__(self, run_config: RunConfig, device: DeviceInfo, num_examples: int) -> None:
        missing = [pkg for pkg in _REQUIRED if importlib.util.find_spec(pkg) is None]
        if missing:
            raise DependencyMissingError(
                package="mlx-lm",
                reason="training on Apple Silicon",
                extra_notes=["Install the MLX extra: pip install 'flameforge[mlx]'."],
            )
        super().__init__(run_config, device, num_examples)

    def _write_jsonl(self, directory: Path) -> None:
        """Write mlx-lm's expected ``train.jsonl``/``valid.jsonl`` files."""
        if self.data is None:  # pragma: no cover - guarded by train()
            raise TrainingError(message="Training data was not prepared.")
        (directory / "train.jsonl").write_text(
            "\n".join(json.dumps({"text": t}) for t in self.data.train_texts),
            encoding="utf-8",
        )
        eval_texts = self.data.eval_texts or self.data.train_texts[:1]
        (directory / "valid.jsonl").write_text(
            "\n".join(json.dumps({"text": t}) for t in eval_texts),
            encoding="utf-8",
        )

    def train(self) -> Iterator[TrainingMetrics]:
        """Run LoRA training via mlx-lm, yielding metrics per reported step.

        Yields:
            One :class:`TrainingMetrics` per training report.

        Raises:
            TrainingError: If data is missing or the loop fails.
        """
        if self.data is None:
            raise TrainingError(
                message="Training data was not prepared before starting.",
                suggestions=["This is an internal error; please re-run the data step."],
            )
        if self.run_config.method == FineTuningMethod.FULL:
            raise TrainingError(
                message="Full fine-tuning is not supported on the MLX backend yet.",
                suggestions=["Choose LoRA or QLoRA, which MLX supports efficiently."],
            )

        stream = MetricStream()
        worker = threading.Thread(
            target=self._run_loop,
            args=(stream,),
            name="flameforge-mlx-train",
            daemon=True,
        )
        worker.start()

        last: TrainingMetrics | None = None
        for metrics in stream:
            last = metrics
            yield metrics
        worker.join(timeout=5)
        self._finalize(last)

    def _run_loop(self, stream: MetricStream) -> None:
        """Body of the MLX training thread; pushes metrics into ``stream``."""
        try:
            import mlx.optimizers as optim
            from mlx_lm import load
            from mlx_lm.tuner.trainer import TrainingArgs, train
            from mlx_lm.tuner.utils import linear_to_lora_layers

            cfg = self.run_config.training
            model, tokenizer = load(self.run_config.model_id)
            model.freeze()
            linear_to_lora_layers(
                model,
                num_layers=16,
                config={
                    "rank": cfg.lora_rank,
                    "alpha": cfg.lora_alpha,
                    "dropout": cfg.lora_dropout,
                    "scale": cfg.lora_alpha / cfg.lora_rank,
                },
            )

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                self._write_jsonl(tmp_path)
                from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset

                train_set, valid_set, _ = mlx_load_dataset(tmp_path, tokenizer)
                args = TrainingArgs(
                    batch_size=cfg.per_device_batch_size,
                    iters=self.total_steps,
                    val_batches=1,
                    steps_per_report=1,
                    steps_per_eval=cfg.save_steps,
                    adapter_file=str(Path(cfg.output_dir) / "adapters.safetensors"),
                    max_seq_length=cfg.max_seq_length,
                )
                callback = _MlxCallback(stream, self)
                Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
                train(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optim.Adam(learning_rate=cfg.learning_rate),
                    train_dataset=train_set,
                    val_dataset=valid_set,
                    args=args,
                    training_callback=callback,
                )
            stream.finish()
        except Exception as exc:  # surface any failure to the consumer cleanly
            _log.exception("MLX training failed")
            stream.finish(TrainingError(message=f"MLX training failed: {exc}", details=str(exc)))

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
        """Request a checkpoint at the next reporting boundary."""
        self.control.checkpoint_requested.set()
        return Path(self.run_config.training.output_dir)


class _MlxCallback:
    """Translates mlx-lm tuner reports into FlameForge metrics."""

    def __init__(self, stream: MetricStream, trainer: MlxTrainer) -> None:
        self._stream = stream
        self._trainer = trainer
        self._start = time.monotonic()
        self._best = float("inf")

    def on_train_loss_report(self, info: dict[str, float]) -> None:
        """Handle a periodic training-loss report from the tuner."""
        import mlx.core as mx

        self._trainer.control.wait_if_paused()
        step = int(info.get("iteration", 0))
        loss = float(info.get("train_loss", 0.0))
        self._best = min(self._best, loss)
        elapsed = time.monotonic() - self._start
        per_step = elapsed / max(1, step)
        total = self._trainer.total_steps
        cfg = self._trainer.run_config.training

        mem_used = 0.0
        with contextlib.suppress(Exception):
            mem_used = mx.get_active_memory() / 1e9  # type: ignore[attr-defined]
        if step % 20 == 0:
            with contextlib.suppress(Exception):
                mx.clear_cache()  # type: ignore[attr-defined]

        self._stream.put(
            TrainingMetrics(
                step=step,
                total_steps=total,
                epoch=round(step / max(1, total) * cfg.num_epochs, 3),
                loss=loss,
                learning_rate=cfg.learning_rate,
                grad_norm=0.0,
                tokens_per_sec=float(info.get("tokens_per_second", 0.0)),
                elapsed_sec=elapsed,
                eta_sec=per_step * max(0, total - step),
                mem_used_gb=round(mem_used, 2),
                mem_budget_gb=self._trainer.device.memory_budget_gb,
                best_loss=round(self._best, 4),
            )
        )

    def on_val_loss_report(self, info: dict[str, float]) -> None:
        """Handle a validation-loss report (currently folded into the next step)."""
        # Validation loss is surfaced via the next train report's eval field; the
        # tuner calls this separately, so we simply log it.
        _log.info("MLX val loss: %s", info.get("val_loss"))
