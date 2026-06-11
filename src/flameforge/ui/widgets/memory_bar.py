"""A horizontal memory-usage bar with the budget marked.

Shows used memory as a filled bar against the budget, colouring the readout
amber/red as usage approaches the configured pressure thresholds so the user can
see trouble coming before the system does.
"""

from __future__ import annotations

from textual.widgets import Static

from flameforge.constants import MEMORY_PAUSE_THRESHOLD, MEMORY_REDUCE_BATCH_THRESHOLD

_BAR_WIDTH = 24


def render_memory_bar(used_gb: float, budget_gb: float, width: int = _BAR_WIDTH) -> str:
    """Render a textual memory bar like ``9.8/12.0 GB ████████░``.

    Args:
        used_gb: Memory currently in use.
        budget_gb: The memory budget.
        width: Bar width in characters.

    Returns:
        A single-line string showing usage and a filled bar.
    """
    budget = max(budget_gb, 0.001)
    frac = min(1.0, max(0.0, used_gb / budget))
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{used_gb:.1f}/{budget_gb:.1f} GB  {bar}  {frac * 100:.0f}%"


class MemoryBar(Static):
    """Live memory-usage bar that changes colour under pressure."""

    def __init__(self) -> None:
        super().__init__("", id="memory-bar")

    def update_usage(self, used_gb: float, budget_gb: float) -> None:
        """Update the bar and its severity colour.

        Args:
            used_gb: Memory currently in use.
            budget_gb: The memory budget.
        """
        self.update(render_memory_bar(used_gb, budget_gb))
        frac = used_gb / budget_gb if budget_gb else 0.0
        self.set_class(frac >= MEMORY_REDUCE_BATCH_THRESHOLD and frac < MEMORY_PAUSE_THRESHOLD, "warn")
        self.set_class(frac >= MEMORY_PAUSE_THRESHOLD, "error")
