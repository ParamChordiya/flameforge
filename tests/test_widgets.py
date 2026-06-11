"""Tests for the pure rendering functions behind the dashboard widgets."""

from __future__ import annotations

from flameforge.ui.widgets.loss_chart import render_line_chart
from flameforge.ui.widgets.memory_bar import render_memory_bar
from flameforge.ui.widgets.stage_bar import render_stage_bar


def test_loss_chart_empty() -> None:
    assert "Waiting" in render_line_chart([])


def test_loss_chart_renders_axis_and_marks() -> None:
    chart = render_line_chart([2.5, 2.0, 1.5, 1.0, 0.5], width=20, height=6)
    lines = chart.splitlines()
    assert len(lines) == 7  # height rows + baseline
    assert "●" in chart
    # Top label is the max, bottom label the min.
    assert "2.50" in lines[0]
    assert "0.50" in lines[5]


def test_loss_chart_downsamples_long_series() -> None:
    chart = render_line_chart([float(i) for i in range(1000)], width=30, height=8)
    # No data row should be wider than the requested width.
    for line in chart.splitlines()[:-1]:
        body = line.split("│", 1)[1] if "│" in line else ""
        assert len(body) <= 30


def test_loss_chart_constant_series() -> None:
    # A flat series must not divide by zero.
    chart = render_line_chart([1.0, 1.0, 1.0], width=10, height=5)
    assert "●" in chart


def test_memory_bar_fractions() -> None:
    bar = render_memory_bar(6.0, 12.0, width=10)
    assert "6.0/12.0 GB" in bar
    assert "50%" in bar
    assert "█" in bar and "░" in bar


def test_memory_bar_full_and_empty() -> None:
    assert "100%" in render_memory_bar(20.0, 12.0, width=10)  # clamped
    assert "0%" in render_memory_bar(0.0, 12.0, width=10)


def test_stage_bar_highlights_current() -> None:
    bar = render_stage_bar(5)
    assert "▶" in bar and "◀" in bar
    assert "Training" in bar
    # Earlier stages are checked off.
    assert "Model ✓" in bar
