"""A real-time ASCII line chart for the training loss curve.

The chart keeps a rolling history of loss values and renders them into a compact
grid with y-axis labels. It is deliberately dependency-free (no plotting library)
so it works in any terminal, and it downsamples to the available width so long
runs still render smoothly.
"""

from __future__ import annotations

from textual.widgets import Static

_MARK = "●"
_DEFAULT_WIDTH = 58
_DEFAULT_HEIGHT = 9


def render_line_chart(values: list[float], width: int = _DEFAULT_WIDTH, height: int = _DEFAULT_HEIGHT) -> str:
    """Render a list of values into a multi-line ASCII line chart.

    Args:
        values: The series to plot (e.g. loss per step). Empty yields a hint.
        width: Plot width in columns (excluding the y-axis gutter).
        height: Plot height in rows.

    Returns:
        A multi-line string with y-axis labels and an x-axis baseline.
    """
    if not values:
        return "Waiting for first step…"

    # Downsample (bucket-average) to at most `width` columns.
    cols = _downsample(values, width)
    lo = min(cols)
    hi = max(cols)
    if hi == lo:
        hi = lo + 1.0  # avoid a zero-height range

    grid = [[" " for _ in range(len(cols))] for _ in range(height)]
    for x, value in enumerate(cols):
        # Row 0 is the top (highest value); invert so larger loss is higher up.
        frac = (value - lo) / (hi - lo)
        row = int(round((1 - frac) * (height - 1)))
        grid[row][x] = _MARK

    gutter = 6
    lines: list[str] = []
    for r in range(height):
        if r == 0:
            label = f"{hi:>5.2f}"
        elif r == height - 1:
            label = f"{lo:>5.2f}"
        else:
            label = " " * 5
        lines.append(f"{label} │{''.join(grid[r])}")
    lines.append(" " * gutter + "└" + "─" * len(cols))
    return "\n".join(lines)


def _downsample(values: list[float], width: int) -> list[float]:
    """Bucket-average ``values`` down to at most ``width`` points."""
    n = len(values)
    if n <= width:
        return list(values)
    bucket = n / width
    out: list[float] = []
    for i in range(width):
        start = int(i * bucket)
        end = int((i + 1) * bucket) or start + 1
        chunk = values[start:end] or [values[start]]
        out.append(sum(chunk) / len(chunk))
    return out


class LossChart(Static):
    """A Static widget that plots a live-updating loss curve."""

    def __init__(self, max_points: int = 2000) -> None:
        super().__init__("Waiting for first step…", id="loss-chart")
        self._values: list[float] = []
        self._max_points = max_points

    def add_point(self, loss: float) -> None:
        """Append a loss value and re-render the chart.

        Args:
            loss: The latest training loss.
        """
        self.add_points([loss])

    def add_points(self, losses: list[float]) -> None:
        """Append several loss values and re-render once.

        Args:
            losses: New loss values to append. No-op if empty.
        """
        if not losses:
            return
        self._values.extend(losses)
        if len(self._values) > self._max_points:
            self._values = self._values[-self._max_points :]
        self.update(render_line_chart(self._values))

    def reset(self) -> None:
        """Clear the chart back to its empty state."""
        self._values.clear()
        self.update("Waiting for first step…")
