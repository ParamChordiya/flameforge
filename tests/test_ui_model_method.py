"""Async UI tests for the model- and method-selection screens."""

from __future__ import annotations

import pytest
from textual.widgets import Button, DataTable

import flameforge.ui.screens.model_select as model_select
from flameforge.app import FlameForgeApp
from flameforge.constants import Backend, DeviceType, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.models.registry import get_model


def _device() -> DeviceInfo:
    return DeviceInfo(DeviceType.MPS, "Test M", 16.0, 11.2, None, 1, Backend.MLX)


@pytest.mark.asyncio
async def test_popular_table_populated() -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.advance_to("model_select")
        await pilot.pause()
        table = app.screen.query_one("#popular-table", DataTable)
        assert table.row_count >= 15
        assert "Fits" in [str(c.label) for c in table.columns.values()]


@pytest.mark.asyncio
async def test_choose_non_gated_advances_to_method() -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.advance_to("model_select")
        await pilot.pause()
        app.screen._choose_registry_model(get_model("Qwen/Qwen2.5-3B-Instruct"))
        await pilot.pause()
        assert type(app.screen).__name__ == "MethodSelectScreen"
        assert app.session.model_id == "Qwen/Qwen2.5-3B-Instruct"


@pytest.mark.asyncio
async def test_method_selection_advances_to_data() -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.model_id = "Qwen/Qwen2.5-3B-Instruct"
        app.session.model_info = get_model("Qwen/Qwen2.5-3B-Instruct")
        app.advance_to("method_select")
        await pilot.pause()
        # The recommended method for a 3B model on an 11 GB budget should fit.
        assert app.screen._recommended_method() in {FineTuningMethod.QLORA, FineTuningMethod.LORA}
        button = next(b for b in app.screen.query(Button) if b.id == "select-lora")
        app.screen.on_button_pressed(Button.Pressed(button))
        await pilot.pause()
        assert app.session.method == FineTuningMethod.LORA
        assert type(app.screen).__name__ == "DataLoadScreen"


@pytest.mark.asyncio
async def test_gated_model_without_token_opens_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_select, "find_hf_token", lambda: None)
    app = FlameForgeApp(device=_device())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.advance_to("model_select")
        await pilot.pause()
        app.screen._choose_registry_model(get_model("meta-llama/Llama-3.1-8B-Instruct"))
        await pilot.pause()
        assert type(app.screen).__name__ == "AuthScreen"
        app.screen.action_cancel()
        await pilot.pause()
        assert type(app.screen).__name__ == "ModelSelectScreen"


@pytest.mark.asyncio
async def test_full_finetune_disabled_for_large_model() -> None:
    app = FlameForgeApp(device=_device())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        app.session.model_info = get_model("meta-llama/Meta-Llama-3.1-70B-Instruct")
        app.advance_to("method_select")
        await pilot.pause()
        full_btn = next(b for b in app.screen.query(Button) if b.id == "select-full")
        assert full_btn.disabled is True
