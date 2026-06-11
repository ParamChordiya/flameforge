"""A pipeline stage indicator showing progress through the six steps."""

from __future__ import annotations

from textual.widgets import Static

_STAGES = ["Model", "Method", "Data", "Config", "Training", "Export"]


def render_stage_bar(current: int) -> str:
    """Render the stage breadcrumb, highlighting the current step.

    Args:
        current: The 1-based index of the active stage.

    Returns:
        A single line like ``① Model ▸ ② Method ▸ ... ▶ ⑤ Training ◀ ...``.
    """
    circled = "①②③④⑤⑥"
    parts: list[str] = []
    for i, name in enumerate(_STAGES, start=1):
        glyph = circled[i - 1]
        if i == current:
            parts.append(f"▶ {glyph} {name} ◀")
        elif i < current:
            parts.append(f"{glyph} {name} ✓")
        else:
            parts.append(f"{glyph} {name}")
    return "   ".join(parts)


class StageBar(Static):
    """A breadcrumb widget for the active pipeline stage."""

    def __init__(self, current: int = 1) -> None:
        super().__init__(render_stage_bar(current), id="stage-bar-widget")

    def set_stage(self, current: int) -> None:
        """Update which stage is highlighted.

        Args:
            current: The 1-based active stage index.
        """
        self.update(render_stage_bar(current))
