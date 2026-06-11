"""Async UI tests for the export screen."""

from __future__ import annotations

from pathlib import Path

import pytest

from flameforge.app import FlameForgeApp
from flameforge.constants import Backend, DeviceType, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.models.registry import get_model
from flameforge.training.base import TrainingResult


def _device() -> DeviceInfo:
    return DeviceInfo(DeviceType.MPS, "M3", 18.0, 12.6, None, 1, Backend.MLX)


def _seed(app: FlameForgeApp, tmp: Path, simulated: bool) -> None:
    s = app.session
    s.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    s.model_info = get_model(s.model_id)
    s.method = FineTuningMethod.LORA
    s.output_dir = tmp
    s.training_result = TrainingResult(
        final_loss=0.74,
        best_loss=0.71,
        steps_completed=10,
        elapsed_sec=42.0,
        total_tokens=12345,
        output_dir=tmp,
        stopped_early=False,
        simulated=simulated,
    )


@pytest.mark.asyncio
async def test_export_summary_and_adapter(tmp_path: Path) -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        _seed(app, tmp_path, simulated=True)
        app.advance_to("export")
        await pilot.pause()
        screen = app.screen
        assert "0.74" in str(screen.query_one("#export-summary").render())
        screen._export_adapter()
        await pilot.pause()
        assert "Adapter ready" in str(screen.query_one("#export-status").render())


@pytest.mark.asyncio
async def test_export_merge_missing_adapter_shows_error(tmp_path: Path) -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        # Point at an empty dir so the merge has no adapter to work with.
        _seed(app, tmp_path / "empty", simulated=False)
        app.advance_to("export")
        await pilot.pause()
        screen = app.screen
        screen._run_export("merged")
        for _ in range(40):
            await pilot.pause(0.05)
            if "✗" in str(screen.query_one("#export-status").render()):
                break
        assert "✗" in str(screen.query_one("#export-status").render())
