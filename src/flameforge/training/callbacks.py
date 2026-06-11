"""Plumbing that bridges backend training loops to the TUI's metrics stream.

Real backends run their training loop on a worker thread and push
:class:`~flameforge.training.base.TrainingMetrics` into a :class:`MetricStream`;
the trainer's ``train()`` generator simply iterates that stream. For the CUDA
backend (HuggingFace TRL/transformers), :func:`build_hf_callback` lazily
constructs a ``TrainerCallback`` that converts log events into metrics and honours
the shared :class:`~flameforge.training.base.TrainerControl` flags.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from typing import Any

from flameforge.device.detector import DeviceInfo
from flameforge.device.memory import current_process_memory_gb
from flameforge.training.base import TrainerControl, TrainingMetrics
from flameforge.utils.logging import get_logger

_log = get_logger("training.callbacks")


class MetricStream:
    """A thread-safe, iterable stream of training metrics with a done sentinel.

    A producer thread calls :meth:`put` for each logged step and :meth:`finish`
    when training ends (optionally with an error). A single consumer iterates the
    stream, which blocks until the next item arrives or the stream is finished.
    """

    _SENTINEL = object()

    def __init__(self) -> None:
        self._q: queue.Queue[object] = queue.Queue()
        self._error: BaseException | None = None

    def put(self, metrics: TrainingMetrics) -> None:
        """Enqueue one metrics snapshot."""
        self._q.put(metrics)

    def finish(self, error: BaseException | None = None) -> None:
        """Signal that no more metrics will be produced.

        Args:
            error: An exception to re-raise on the consumer side, if training
                failed.
        """
        self._error = error
        self._q.put(self._SENTINEL)

    def __iter__(self) -> Iterator[TrainingMetrics]:
        """Yield metrics until the stream is finished, re-raising any error."""
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            assert isinstance(item, TrainingMetrics)
            yield item


def build_hf_callback(
    stream: MetricStream,
    control: TrainerControl,
    device: DeviceInfo,
    total_steps: int,
) -> object:
    """Construct a transformers ``TrainerCallback`` feeding ``stream``.

    The class is defined inside the function so ``transformers`` is only imported
    on machines that actually train on CUDA.

    Args:
        stream: The metric stream to push snapshots into.
        control: Shared control flags (pause/stop/checkpoint).
        device: The detected device (for memory reporting).
        total_steps: Total planned steps (for ETA/progress).

    Returns:
        A ``TrainerCallback`` instance.

    Raises:
        ImportError: If transformers is not installed (handled by the caller).
    """
    from transformers import TrainerCallback  # type: ignore[import-not-found]

    class _FlameForgeCallback(TrainerCallback):  # type: ignore[misc]
        """Translates transformers training events into FlameForge metrics."""

        def __init__(self) -> None:
            self._start = time.monotonic()
            self._best = float("inf")
            self._last_ckpt: int | None = None

        def on_log(
            self, args: Any, state: Any, control_obj: Any, logs: dict[str, float] | None = None, **_: object
        ) -> Any:
            """Convert a transformers log dict into a TrainingMetrics snapshot."""
            if not logs or "loss" not in logs:
                return control_obj
            step = int(getattr(state, "global_step", 0))
            loss = float(logs.get("loss", 0.0))
            self._best = min(self._best, loss)
            elapsed = time.monotonic() - self._start
            per_step = elapsed / max(1, step)
            stream.put(
                TrainingMetrics(
                    step=step,
                    total_steps=total_steps,
                    epoch=float(getattr(state, "epoch", 0.0) or 0.0),
                    loss=loss,
                    learning_rate=float(logs.get("learning_rate", 0.0)),
                    grad_norm=float(logs.get("grad_norm", 0.0)),
                    tokens_per_sec=float(logs.get("train_tokens_per_second", 0.0)),
                    elapsed_sec=elapsed,
                    eta_sec=per_step * max(0, total_steps - step),
                    mem_used_gb=current_process_memory_gb(),
                    mem_budget_gb=device.memory_budget_gb,
                    eval_loss=float(logs["eval_loss"]) if "eval_loss" in logs else None,
                    best_loss=round(self._best, 4),
                    last_checkpoint_step=self._last_ckpt,
                )
            )
            return control_obj

        def on_save(self, args: Any, state: Any, control_obj: Any, **_: object) -> Any:
            """Record the step of the most recent checkpoint."""
            self._last_ckpt = int(getattr(state, "global_step", 0))
            return control_obj

        def on_step_end(self, args: Any, state: Any, control_obj: Any, **_: object) -> Any:
            """Honour pause/stop requests between steps."""
            control.wait_if_paused()
            if control.stop_requested.is_set():
                control_obj.should_training_stop = True
            if control.checkpoint_requested.is_set():
                control_obj.should_save = True
                control.checkpoint_requested.clear()
            return control_obj

    return _FlameForgeCallback()
