"""The FlameForge Textual application — the entry point for the TUI.

The app owns a single :class:`~flameforge.ui.state.SessionState` and drives a
linear screen flow: welcome → model → method → data → config → training →
export. Screens advance via :meth:`FlameForgeApp.advance_to`, which fails
gracefully (with a toast) if a target screen is not yet registered, so the app is
always launchable while under construction.
"""

from __future__ import annotations

from typing import Any

from textual.app import App
from textual.screen import Screen

from flameforge.device.detector import DeviceInfo, detect_device
from flameforge.ui.state import SessionState
from flameforge.utils.logging import get_logger


class FlameForgeApp(App[None]):
    """Main Textual application for FlameForge.

    Args:
        device: Pre-detected device info. If None, detection runs at startup.
            Injecting it makes the app trivially testable.
    """

    TITLE = "FlameForge"
    SUB_TITLE = "LLM Fine-Tuning Made Simple"
    CSS_PATH = "ui/styles/app.tcss"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, device: DeviceInfo | None = None) -> None:
        super().__init__()
        self._file_logger = get_logger("app")
        self._device = device or detect_device()
        self.session = SessionState(device=self._device)
        self._registered: dict[str, type[Screen[Any]]] = {}

    def on_mount(self) -> None:
        """Register available screens and show the welcome screen."""
        self._registered = _build_screen_registry()
        self._file_logger.info("Registered screens: %s", ", ".join(self._registered))
        self.push_screen(self._registered["welcome"]())

    def advance_to(self, screen_name: str) -> None:
        """Navigate forward to a named screen, or toast if it is unavailable.

        A fresh screen instance is pushed each time so screens always re-read the
        latest :class:`SessionState`.

        Args:
            screen_name: The registered name of the target screen.
        """
        screen_cls = self._registered.get(screen_name)
        if screen_cls is not None:
            self.push_screen(screen_cls())
        else:
            self.notify(
                f"The '{screen_name}' step isn't available in this build yet.",
                title="Coming soon",
                severity="warning",
            )

    def go_back(self) -> None:
        """Pop the current screen, returning to the previous step if possible."""
        if len(self.screen_stack) > 1:
            self.pop_screen()


def _import_or_none(module: str, attr: str) -> type[Screen[Any]] | None:
    """Import a screen class, returning None if it is not yet present.

    Args:
        module: Dotted module path.
        attr: The screen class name within that module.

    Returns:
        The screen class, or None if the module/attribute does not exist.
    """
    try:
        mod = __import__(module, fromlist=[attr])
        obj = getattr(mod, attr)
    except (ImportError, AttributeError):
        return None
    return obj if isinstance(obj, type) and issubclass(obj, Screen) else None


def _build_screen_registry() -> dict[str, type[Screen[Any]]]:
    """Discover and return all implemented screens keyed by their flow name.

    The welcome screen is always present; later screens are imported defensively
    so the app remains launchable while the pipeline is still being built.

    Returns:
        A mapping of flow name to screen class for every available screen.
    """
    from flameforge.ui.screens.welcome import WelcomeScreen

    registry: dict[str, type[Screen[Any]]] = {"welcome": WelcomeScreen}
    optional = {
        "model_select": ("flameforge.ui.screens.model_select", "ModelSelectScreen"),
        "method_select": ("flameforge.ui.screens.method_select", "MethodSelectScreen"),
        "data_load": ("flameforge.ui.screens.data_load", "DataLoadScreen"),
        "config_edit": ("flameforge.ui.screens.config_edit", "ConfigEditScreen"),
        "training": ("flameforge.ui.screens.training", "TrainingScreen"),
        "export": ("flameforge.ui.screens.export", "ExportScreen"),
    }
    for name, (module, attr) in optional.items():
        screen_cls = _import_or_none(module, attr)
        if screen_cls is not None:
            registry[name] = screen_cls
    return registry
