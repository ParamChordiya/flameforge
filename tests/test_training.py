"""Tests for the trainer interface, simulated trainer, and metric stream."""

from __future__ import annotations

from pathlib import Path

import pytest

from flameforge.config import DataConfig, RunConfig, TrainingConfig
from flameforge.constants import Backend, DataFormat, DeviceType, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.training.base import (
    SimulatedTrainer,
    TrainingMetrics,
    build_trainer,
)
from flameforge.training.callbacks import MetricStream


def _device() -> DeviceInfo:
    return DeviceInfo(DeviceType.MPS, "M", 18.0, 12.6, None, 1, Backend.MLX)


def _run_config(alpaca_file: Path, **train_overrides: object) -> RunConfig:
    data = DataConfig(path=alpaca_file, data_format=DataFormat.ALPACA, num_examples=20)
    training = TrainingConfig(num_epochs=2, effective_batch_size=4, save_steps=3, **train_overrides)
    return RunConfig(model_id="x/y", method=FineTuningMethod.LORA, data=data, training=training)


def test_total_steps_math(alpaca_file: Path) -> None:
    trainer = SimulatedTrainer(_run_config(alpaca_file), _device(), num_examples=20)
    # ceil(20/4) * 2 epochs = 10
    assert trainer.total_steps == 10


def test_simulated_loss_decreases(alpaca_file: Path) -> None:
    trainer = SimulatedTrainer(_run_config(alpaca_file), _device(), num_examples=40, step_delay=0.0)
    losses = [m.loss for m in trainer.train()]
    assert len(losses) == trainer.total_steps
    assert losses[0] > losses[-1]
    assert trainer.result is not None
    assert trainer.result.simulated is True
    assert trainer.result.steps_completed == trainer.total_steps


def test_simulated_stop_early(alpaca_file: Path) -> None:
    trainer = SimulatedTrainer(_run_config(alpaca_file), _device(), num_examples=200, step_delay=0.0)
    gen = trainer.train()
    next(gen)
    next(gen)
    trainer.control.stop_requested.set()
    list(gen)  # drain
    assert trainer.result is not None
    assert trainer.result.stopped_early is True
    assert trainer.result.steps_completed < trainer.total_steps


def test_build_trainer_falls_back_to_simulator(alpaca_file: Path) -> None:
    # Neither mlx-lm nor trl is installed in CI → simulator.
    trainer = build_trainer(_run_config(alpaca_file), _device(), 20)
    assert trainer.simulated is True


def test_metrics_progress_fraction() -> None:
    m = TrainingMetrics(
        step=5,
        total_steps=10,
        epoch=1.0,
        loss=1.0,
        learning_rate=1e-4,
        grad_norm=0.5,
        tokens_per_sec=100.0,
        elapsed_sec=1.0,
        eta_sec=1.0,
        mem_used_gb=4.0,
        mem_budget_gb=8.0,
    )
    assert m.progress_fraction == pytest.approx(0.5)


def test_metric_stream_roundtrip() -> None:
    stream = MetricStream()
    sample = TrainingMetrics(1, 2, 0.5, 1.0, 1e-4, 0.1, 10.0, 0.1, 0.1, 1.0, 2.0)
    stream.put(sample)
    stream.finish()
    received = list(stream)
    assert received == [sample]


def test_metric_stream_reraises_error() -> None:
    stream = MetricStream()
    stream.finish(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        list(stream)


def test_checkpoint_returns_path(alpaca_file: Path) -> None:
    trainer = SimulatedTrainer(_run_config(alpaca_file, output_dir="/tmp/ff-out"), _device(), 20)
    path = trainer.save_checkpoint(100)
    assert path == Path("/tmp/ff-out") / "checkpoint-100"
