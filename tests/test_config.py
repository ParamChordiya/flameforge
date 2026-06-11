"""Tests for the Pydantic configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameforge.config import DataConfig, RunConfig, TrainingConfig
from flameforge.constants import DataFormat, FineTuningMethod
from flameforge.utils.errors import ConfigurationError


def test_training_config_defaults() -> None:
    cfg = TrainingConfig()
    assert cfg.num_epochs == 3
    assert cfg.learning_rate == pytest.approx(2.0e-4)
    assert cfg.lr_scheduler == "cosine"
    assert cfg.effective_batch() == cfg.per_device_batch_size * cfg.gradient_accumulation


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", -1.0),
        ("learning_rate", 0.0),
        ("num_epochs", 0),
        ("lora_dropout", 1.5),
        ("train_eval_split", 1.0),
        ("train_eval_split", 0.0),
        ("warmup_ratio", 0.9),
    ],
)
def test_training_config_rejects_invalid(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(**{field: value})


def test_training_config_rejects_unknown_scheduler() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(lr_scheduler="magic")


def test_training_config_rejects_both_precisions() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(bf16=True, fp16=True)


def test_training_config_validate_assignment() -> None:
    cfg = TrainingConfig()
    with pytest.raises(ValidationError):
        cfg.learning_rate = -5.0


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("num_epochs: 5\nlearning_rate: 1.0e-4\nlora_rank: 8\n", encoding="utf-8")
    cfg = TrainingConfig.from_yaml(yaml_path)
    assert cfg.num_epochs == 5
    assert cfg.learning_rate == pytest.approx(1.0e-4)
    assert cfg.lora_rank == 8


def test_from_yaml_missing_file_raises_friendly(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        TrainingConfig.from_yaml(tmp_path / "nope.yaml")


def test_from_yaml_invalid_values_raise_friendly(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("num_epochs: -3\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        TrainingConfig.from_yaml(yaml_path)


def test_data_config_requires_existing_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(path=tmp_path / "missing.jsonl", data_format=DataFormat.ALPACA)


def test_run_config_uses_adapter(alpaca_file: Path) -> None:
    data = DataConfig(path=alpaca_file, data_format=DataFormat.ALPACA, num_examples=3)
    run = RunConfig(model_id="x/y", method=FineTuningMethod.LORA, data=data)
    assert run.uses_adapter is True
    run.method = FineTuningMethod.FULL
    assert run.uses_adapter is False
