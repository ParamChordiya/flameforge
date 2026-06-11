"""Async UI tests for the config-edit and training-dashboard screens."""

from __future__ import annotations

from pathlib import Path

import pytest

from flameforge.app import FlameForgeApp
from flameforge.config import DataConfig
from flameforge.constants import Backend, DataFormat, DeviceType, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.models.registry import get_model
from flameforge.ui.widgets.loss_chart import LossChart


def _device() -> DeviceInfo:
    return DeviceInfo(DeviceType.MPS, "M3", 18.0, 12.6, None, 1, Backend.MLX)


def _seed(app: FlameForgeApp, alpaca_file: Path) -> None:
    s = app.session
    s.model_id = "Qwen/Qwen2.5-3B-Instruct"
    s.model_info = get_model(s.model_id)
    s.method = FineTuningMethod.LORA
    s.data = DataConfig(path=alpaca_file, data_format=DataFormat.ALPACA, num_examples=3)
    s.training = s.training.model_copy(update={"num_epochs": 2, "effective_batch_size": 2, "save_steps": 2})


@pytest.mark.asyncio
async def test_config_screen_autotunes_and_starts(alpaca_file: Path) -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        _seed(app, alpaca_file)
        app.advance_to("config_edit")
        await pilot.pause()
        # Small dataset → auto-tune should have reduced epochs to 1.
        assert app.session.training.num_epochs == 1
        app.screen._start()
        await pilot.pause()
        assert type(app.screen).__name__ == "TrainingScreen"


@pytest.mark.asyncio
async def test_config_screen_rejects_bad_value(alpaca_file: Path) -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        _seed(app, alpaca_file)
        app.advance_to("config_edit")
        await pilot.pause()
        app.screen.query_one("#cfg-num_epochs").value = "not-a-number"
        error_widget = app.screen.query_one("#config-error")
        app.screen._start()
        await pilot.pause()
        # Stays on the config screen and shows an error.
        assert type(app.screen).__name__ == "ConfigEditScreen"
        assert "✗" in str(error_widget.render())


@pytest.mark.asyncio
async def test_training_dashboard_runs_to_completion(alpaca_file: Path) -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        _seed(app, alpaca_file)
        app.advance_to("training")
        await pilot.pause()
        screen = app.screen
        assert screen._trainer is not None and screen._trainer.simulated
        for _ in range(200):
            await pilot.pause(0.05)
            if screen._done:
                break
        assert screen._done
        assert app.session.training_result is not None
        assert app.session.training_result.steps_completed > 0
        assert screen.query_one("#export-btn").disabled is False
        assert len(screen.query_one(LossChart)._values) > 0
