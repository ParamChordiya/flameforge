"""Welcome screen: ASCII logo, detected device, and a Start action.

This is the first screen the user sees. It confirms the auto-detected device and
memory budget so there are no surprises later, lets the user tune the memory
budget (especially important on Apple Silicon, where GPU memory is shared with
macOS), and warns prominently if we have fallen back to CPU-only execution.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Static

from flameforge import __version__
from flameforge.constants import DeviceType
from flameforge.device.detector import with_memory_cap
from flameforge.device.memory import default_memory_budget, exceeds_safe_budget

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
            # Let the user set the memory budget to use. On Apple Silicon this is
            # the safety valve that keeps macOS responsive during training.
            mac_note = (
                "  On Apple Silicon, leave headroom for macOS — raising this is risky."
                if device.device_type == DeviceType.MPS
                else ""
            )
            yield Static(
                f"Memory budget to use (GB), of {device.total_memory_gb:.1f} GB total."
                + (f"\n{mac_note}" if mac_note else ""),
                classes="hint",
            )
            with Horizontal(id="mem-row"):
                yield Input(
                    value=f"{device.memory_budget_gb:.1f}",
                    placeholder=f"{device.memory_budget_gb:.1f}",
                    id="mem-budget",
                )
                yield Button("Apply", id="apply-mem")
            yield Static("", id="mem-status")
            with Center(id="welcome-buttons"):
                yield Button("Start  ▶", id="start", variant="primary")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Apply the memory budget when Enter is pressed in the field."""
        if event.input.id == "mem-budget":
            self._apply_memory()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Apply and Start."""
        if event.button.id == "apply-mem":
            self._apply_memory()
        elif event.button.id == "start":
            self.action_start()

    def _apply_memory(self) -> bool:
        """Validate and apply the memory budget from the input.

        Returns:
            True if the value was valid (and applied), False otherwise.
        """
        raw = self.query_one("#mem-budget", Input).value.strip()
        status = self.query_one("#mem-status", Static)
        session = self.app.session  # type: ignore[attr-defined]
        device = session.device
        if not raw:
            # Empty means "use the conservative default".
            session.device = with_memory_cap(device, None)
            self._refresh_device_info()
            status.set_class(False, "error", "warn")
            return True
        try:
            value = float(raw)
        except ValueError:
            status.set_class(True, "error")
            status.set_class(False, "warn")
            status.update(f"✗ '{raw}' is not a number.")
            return False
        if value <= 0 or value > device.total_memory_gb:
            status.set_class(True, "error")
            status.set_class(False, "warn")
            status.update(f"✗ Enter a value between 0 and {device.total_memory_gb:.1f} GB.")
            return False

        session.device = with_memory_cap(device, value)
        self._refresh_device_info()
        if exceeds_safe_budget(device, session.device.memory_budget_gb):
            safe = default_memory_budget(device)
            status.set_class(False, "error")
            status.set_class(True, "warn")
            status.update(
                f"⚠ {session.device.memory_budget_gb:.1f} GB exceeds the safe default of {safe:.1f} GB — "
                "this may cause instability."
            )
        else:
            status.set_class(False, "error", "warn")
            status.update(f"✓ Budget set to {session.device.memory_budget_gb:.1f} GB.")
        return True

    def _refresh_device_info(self) -> None:
        """Re-render the device summary after a budget change."""
        device = self.app.session.device  # type: ignore[attr-defined]
        self.query_one("#device-info", Static).update("\n".join(device.summary_lines()))

    def action_start(self) -> None:
        """Apply any pending memory value, then move to model selection."""
        if not self._apply_memory():
            return
        self.app.advance_to("model_select")  # type: ignore[attr-defined]
