"""Welcome screen: ASCII logo, detected device, and a Start button.

This is the first screen the user sees. It confirms the auto-detected device and
memory budget so there are no surprises later, and warns prominently if we have
fallen back to CPU-only execution.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from flameforge import __version__

_LOGO = r"""
 ______ _                 ______
|  ____| |               |  ____|
| |__  | | __ _ _ __ ___ | |__ ___  _ __ __ _  ___
|  __| | |/ _` | '_ ` _ \|  __/ _ \| '__/ _` |/ _ \
| |    | | (_| | | | | | | | | (_) | | | (_| |  __/
|_|    |_|\__,_|_| |_| |_|_|  \___/|_|  \__, |\___|
                                         __/ |
        Fine-tune any LLM, zero config  |___/
"""


class WelcomeScreen(Screen[None]):
    """Landing screen showing device detection results and a Start action."""

    BINDINGS = [
        ("enter", "start", "Start"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Build the welcome layout."""
        yield Header(show_clock=False)
        device = self.app.session.device  # type: ignore[attr-defined]
        with Center(id="welcome-root"), Vertical(id="welcome-card"):
            yield Static(_LOGO, id="logo")
            yield Static(
                f"Welcome to FlameForge v{__version__}.\nGo from raw data to a fine-tuned model in minutes.",
                classes="subtitle",
            )
            yield Static("\n".join(device.summary_lines()), id="device-info")
            if device.is_cpu_only:
                yield Static(
                    "⚠ No GPU detected — training will run on CPU and be VERY slow.\n"
                    "  This is only practical for the smallest models.",
                    id="cpu-warning",
                )
            with Center(id="welcome-buttons"):
                yield Button("Start  ▶", id="start", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Advance to model selection when Start is pressed."""
        if event.button.id == "start":
            self.action_start()

    def action_start(self) -> None:
        """Move to the model-selection step."""
        self.app.advance_to("model_select")  # type: ignore[attr-defined]
