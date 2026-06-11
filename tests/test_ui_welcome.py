"""Async UI tests for the welcome screen's memory-budget control."""

from __future__ import annotations

import pytest

from flameforge.app import FlameForgeApp
from flameforge.constants import Backend, DeviceType
from flameforge.device.detector import DeviceInfo


def _mac() -> DeviceInfo:
    # 16 GB unified → 11.2 GB safe default.
    return DeviceInfo(DeviceType.MPS, "Apple M2", 16.0, 11.2, None, 1, Backend.MLX)


@pytest.mark.asyncio
async def test_welcome_lowers_budget() -> None:
    app = FlameForgeApp(device=_mac())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#mem-budget").value = "8"
        screen._apply_memory()
        await pilot.pause()
        assert app.session.device.memory_budget_gb == 8.0
        assert "✓" in str(screen.query_one("#mem-status").render())


@pytest.mark.asyncio
async def test_welcome_warns_when_above_safe() -> None:
    app = FlameForgeApp(device=_mac())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#mem-budget").value = "14"
        screen._apply_memory()
        await pilot.pause()
        assert app.session.device.memory_budget_gb == 14.0
        assert "⚠" in str(screen.query_one("#mem-status").render())


@pytest.mark.asyncio
async def test_welcome_rejects_out_of_range_and_blocks_start() -> None:
    app = FlameForgeApp(device=_mac())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#mem-budget").value = "999"  # exceeds 16 GB total
        screen.action_start()
        await pilot.pause()
        # Stays on welcome; budget unchanged.
        assert type(app.screen).__name__ == "WelcomeScreen"
        assert app.session.device.memory_budget_gb == 11.2
        assert "✗" in str(screen.query_one("#mem-status").render())


@pytest.mark.asyncio
async def test_welcome_valid_budget_allows_start() -> None:
    app = FlameForgeApp(device=_mac())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#mem-budget").value = "9.5"
        screen.action_start()
        await pilot.pause()
        assert app.session.device.memory_budget_gb == 9.5
        assert type(app.screen).__name__ == "ModelSelectScreen"
