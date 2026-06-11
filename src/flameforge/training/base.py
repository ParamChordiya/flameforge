"""Abstract trainer interface, live metrics, and a simulated reference trainer.

Every backend (CUDA, MLX) implements :class:`BaseTrainer` and yields
:class:`TrainingMetrics` from :meth:`BaseTrainer.train`. The TUI is completely
backend-agnostic: it iterates the metrics generator, renders them, and toggles
:class:`TrainerControl` flags to pause/stop/checkpoint.

:class:`SimulatedTrainer` produces realistic-looking metrics without any ML
dependencies, so the training dashboard can be exercised (and demoed) on any
machine. :func:`build_trainer` returns a real backend trainer when its libraries
are installed and otherwise falls back to the simulator — the app never crashes
just because a GPU stack is missing.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from flameforge.config import RunConfig
from flameforge.device.detector import DeviceInfo
from flameforge.utils.logging import get_logger

_log = get_logger("training.base")


@dataclass
class TrainingMetrics:
    """A snapshot of training state at one logged step.

    Attributes:
        step: The current global step (1-based).
        total_steps: Total steps planned for the run.
        epoch: Fractional epoch progress.
        loss: The latest training loss.
        learning_rate: The current learning rate.
        grad_norm: The gradient norm at this step.
        tokens_per_sec: Throughput estimate.
        elapsed_sec: Seconds elapsed since training started.
        eta_sec: Estimated seconds remaining.
        mem_used_gb: Memory currently in use.
        mem_budget_gb: The memory budget for this run.
        eval_loss: The most recent eval loss, if one has been computed.
        best_loss: The lowest eval (or train) loss seen so far.
        last_checkpoint_step: The step of the most recent checkpoint, or None.
    """

    step: int
    total_steps: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: float
    tokens_per_sec: float
    elapsed_sec: float
    eta_sec: float
    mem_used_gb: float
    mem_budget_gb: float
    eval_loss: float | None = None
    best_loss: float | None = None
    last_checkpoint_step: int | None = None

    @property
    def progress_fraction(self) -> float:
        """Completion as a fraction in [0, 1]."""
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, self.step / self.total_steps)


@dataclass
class TrainerControl:
    """Thread-safe flags the UI uses to steer a running trainer.

    Attributes:
        stop_requested: Set to ask the trainer to stop early (after a checkpoint).
        checkpoint_requested: Set to ask for a checkpoint at the next step.
    """

    stop_requested: threading.Event = field(default_factory=threading.Event)
    checkpoint_requested: threading.Event = field(default_factory=threading.Event)
    _paused: threading.Event = field(default_factory=threading.Event)

    def pause(self) -> None:
        """Pause training at the next step boundary."""
        self._paused.set()

    def resume(self) -> None:
        """Resume a paused trainer."""
        self._paused.clear()

    @property
    def is_paused(self) -> bool:
        """Whether training is currently paused."""
        return self._paused.is_set()

    def wait_if_paused(self, poll: float = 0.1) -> None:
        """Block while paused, returning promptly if a stop is requested.

        Args:
            poll: How often (seconds) to re-check the pause/stop flags.
        """
        while self._paused.is_set() and not self.stop_requested.is_set():
            time.sleep(poll)


@dataclass
class TrainingData:
    """Formatted, split text ready to hand to a backend trainer.

    Attributes:
        train_texts: Rendered training strings (chat template already applied).
        eval_texts: Rendered evaluation strings.
        tokenizer: Optional loaded tokenizer to reuse during training.
    """

    train_texts: list[str]
    eval_texts: list[str]
    tokenizer: object | None = None


@dataclass
class TrainingResult:
    """Summary returned when training finishes.

    Attributes:
        final_loss: The last training loss observed.
        best_loss: The best (lowest) loss observed.
        steps_completed: How many steps actually ran.
        elapsed_sec: Total wall-clock training time.
        total_tokens: Approximate number of tokens trained on.
        output_dir: Where adapters/checkpoints were written.
        stopped_early: Whether the user stopped before the planned end.
        simulated: Whether this was a simulated (no-backend) run.
    """

    final_loss: float
    best_loss: float
    steps_completed: int
    elapsed_sec: float
    total_tokens: int
    output_dir: Path
    stopped_early: bool
    simulated: bool


class BaseTrainer(ABC):
    """Abstract base class for all training backends.

    Args:
        run_config: The fully assembled run configuration.
        device: The detected device.
        num_examples: Number of training examples (for step math).
    """

    simulated: bool = False

    def __init__(self, run_config: RunConfig, device: DeviceInfo, num_examples: int) -> None:
        self.run_config = run_config
        self.device = device
        self.num_examples = max(1, num_examples)
        self.control = TrainerControl()
        self.result: TrainingResult | None = None
        self.data: TrainingData | None = None

    def set_data(self, data: TrainingData) -> None:
        """Attach the formatted training data a real backend will consume.

        Args:
            data: The prepared, split training/eval text.
        """
        self.data = data

    @property
    def total_steps(self) -> int:
        """Total optimizer steps for the planned run.

        Returns:
            ``ceil(examples / effective_batch) * epochs``, at least 1.
        """
        cfg = self.run_config.training
        steps_per_epoch = math.ceil(self.num_examples / max(1, cfg.effective_batch_size))
        return max(1, steps_per_epoch * cfg.num_epochs)

    @abstractmethod
    def train(self) -> Iterator[TrainingMetrics]:
        """Run training, yielding a :class:`TrainingMetrics` per logged step."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, step: int) -> Path:
        """Persist a checkpoint and return its directory."""
        raise NotImplementedError


class SimulatedTrainer(BaseTrainer):
    """A dependency-free trainer that emits realistic synthetic metrics.

    Used for demos, tests, and graceful fallback when no ML backend is installed.
    The loss follows a decaying exponential with small deterministic noise so the
    dashboard's loss curve looks like a real run.
    """

    simulated = True

    def __init__(
        self,
        run_config: RunConfig,
        device: DeviceInfo,
        num_examples: int,
        step_delay: float = 0.02,
    ) -> None:
        super().__init__(run_config, device, num_examples)
        self._step_delay = step_delay

    def train(self) -> Iterator[TrainingMetrics]:
        """Yield synthetic metrics with a decaying loss until done or stopped."""
        total = self.total_steps
        cfg = self.run_config.training
        start = time.monotonic()
        best = math.inf
        last_ckpt: int | None = None
        approx_tokens_per_step = cfg.effective_batch_size * min(cfg.max_seq_length, 512)

        for step in range(1, total + 1):
            self.control.wait_if_paused()
            if self.control.stop_requested.is_set():
                break

            # Deterministic decaying loss with light, bounded oscillation.
            base_loss = 0.6 + 2.2 * math.exp(-3.0 * step / total)
            noise = 0.04 * math.sin(step * 1.3)
            loss = round(base_loss + noise, 4)
            best = min(best, loss)

            elapsed = time.monotonic() - start
            warmup_steps = max(1, int(cfg.warmup_ratio * total))
            if step <= warmup_steps:
                lr = cfg.learning_rate * step / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total - warmup_steps)
                lr = cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

            if self.control.checkpoint_requested.is_set() or step % cfg.save_steps == 0:
                last_ckpt = step
                self.control.checkpoint_requested.clear()

            per_step = elapsed / step
            yield TrainingMetrics(
                step=step,
                total_steps=total,
                epoch=round(step / total * cfg.num_epochs, 3),
                loss=loss,
                learning_rate=lr,
                grad_norm=round(0.5 + 0.4 * math.exp(-step / total), 3),
                tokens_per_sec=round(approx_tokens_per_step / per_step, 1) if per_step > 0 else 0.0,
                elapsed_sec=elapsed,
                eta_sec=per_step * (total - step),
                mem_used_gb=round(self.device.memory_budget_gb * 0.82, 2),
                mem_budget_gb=self.device.memory_budget_gb,
                eval_loss=round(loss + 0.05, 4) if step % cfg.save_steps == 0 else None,
                best_loss=round(best, 4),
                last_checkpoint_step=last_ckpt,
            )
            time.sleep(self._step_delay)

        elapsed = time.monotonic() - start
        self.result = TrainingResult(
            final_loss=round(best, 4),
            best_loss=round(best, 4),
            steps_completed=min(step, total),
            elapsed_sec=elapsed,
            total_tokens=approx_tokens_per_step * min(step, total),
            output_dir=Path(cfg.output_dir),
            stopped_early=self.control.stop_requested.is_set(),
            simulated=True,
        )

    def save_checkpoint(self, step: int) -> Path:
        """Pretend to save a checkpoint and return its nominal path."""
        path = Path(self.run_config.training.output_dir) / f"checkpoint-{step}"
        _log.info("[simulated] checkpoint at %s", path)
        return path


def build_trainer(run_config: RunConfig, device: DeviceInfo, num_examples: int) -> BaseTrainer:
    """Return the best available trainer for the device, or the simulator.

    A real backend trainer is returned when its libraries import successfully;
    otherwise a :class:`SimulatedTrainer` is returned so the dashboard always
    works. Callers can inspect :attr:`BaseTrainer.simulated` to inform the user.

    Args:
        run_config: The assembled run configuration.
        device: The detected device.
        num_examples: Number of training examples.

    Returns:
        A ready-to-run trainer.
    """
    from flameforge.constants import Backend

    try:
        if device.backend == Backend.MLX:
            from flameforge.training.mlx_trainer import MlxTrainer

            return MlxTrainer(run_config, device, num_examples)
        from flameforge.training.cuda_trainer import CudaTrainer

        return CudaTrainer(run_config, device, num_examples)
    except Exception as exc:  # missing deps or backend init failure → simulate
        _log.warning("Falling back to simulated trainer: %s", exc)
        return SimulatedTrainer(run_config, device, num_examples)
