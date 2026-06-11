"""Async UI tests for the data-loading screen via Textual's pilot harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from flameforge.app import FlameForgeApp
from flameforge.constants import Backend, DeviceType
from flameforge.device.detector import DeviceInfo


def _fake_device() -> DeviceInfo:
    return DeviceInfo(
        device_type=DeviceType.MPS,
        device_name="Test M-Series",
        total_memory_gb=16.0,
        memory_budget_gb=11.2,
        compute_capability=None,
        gpu_count=1,
        backend=Backend.MLX,
    )


async def _load(app: FlameForgeApp, pilot: object, path: Path) -> object:
    """Drive the data screen to load ``path`` and return the screen."""
    app.advance_to("data_load")
    await pilot.pause()  # type: ignore[attr-defined]
    screen = app.screen
    screen.query_one("#data-path").value = str(path)
    screen._begin_load()
    for _ in range(80):
        await pilot.pause(0.05)  # type: ignore[attr-defined]
        if screen.query_one("#data-results").display:
            break
    return screen


@pytest.mark.asyncio
async def test_data_screen_loads_and_confirms(alpaca_file: Path) -> None:
    app = FlameForgeApp(device=_fake_device())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await _load(app, pilot, alpaca_file)
        assert screen.query_one("#data-results").display is True
        assert screen._result.report.is_trainable
        screen._confirm()
        await pilot.pause()
        assert app.session.data is not None
        assert app.session.data.num_examples == 3
        assert app.session.chat_template is not None


@pytest.mark.asyncio
async def test_data_screen_handles_missing_file() -> None:
    app = FlameForgeApp(device=_fake_device())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.advance_to("data_load")
        await pilot.pause()
        screen = app.screen
        screen.query_one("#data-path").value = "/no/such/file.jsonl"
        screen._begin_load()
        await pilot.pause()
        status = screen.query_one("#load-status")
        assert status.has_class("error")
        assert app.session.data is None


@pytest.mark.asyncio
async def test_data_screen_blocks_confirm_on_empty_outputs(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"instruction": "q", "output": ""}\n', encoding="utf-8")
    app = FlameForgeApp(device=_fake_device())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await _load(app, pilot, bad)
        # All outputs empty -> not trainable -> confirm disabled, session untouched.
        assert screen.query_one("#confirm-data").disabled is True
        screen._confirm()
        await pilot.pause()
        assert app.session.data is None
