"""A widget that previews formatted training samples with prev/next navigation.

The preview shows each example exactly as it will be fed to the model (chat
template applied), along with its token count, so users can confirm their data is
being interpreted correctly before committing to a run.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


class DataPreview(Vertical):
    """Displays one formatted sample at a time with navigation controls.

    Args:
        samples: Pairs of (rendered text, token count) to page through.
    """

    def __init__(self, samples: list[tuple[str, int]] | None = None) -> None:
        super().__init__()
        self._samples: list[tuple[str, int]] = samples or []
        self._index = 0

    def compose(self) -> ComposeResult:
        """Build the preview body and navigation buttons."""
        yield Static(id="preview-header", classes="title")
        yield Static(id="preview-body", classes="panel")
        with Horizontal(id="preview-nav"):
            yield Button("◀ Prev", id="preview-prev")
            yield Button("Next ▶", id="preview-next")

    def on_mount(self) -> None:
        """Render the first sample once mounted."""
        self._refresh_view()

    def set_samples(self, samples: list[tuple[str, int]]) -> None:
        """Replace the previewed samples and reset to the first one.

        Args:
            samples: New (rendered text, token count) pairs.
        """
        self._samples = samples
        self._index = 0
        self._refresh_view()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle prev/next navigation."""
        if event.button.id == "preview-prev":
            self._step(-1)
            event.stop()
        elif event.button.id == "preview-next":
            self._step(1)
            event.stop()

    def _step(self, delta: int) -> None:
        """Move the current index by ``delta``, clamped to valid bounds."""
        if not self._samples:
            return
        self._index = max(0, min(len(self._samples) - 1, self._index + delta))
        self._refresh_view()

    def _refresh_view(self) -> None:
        """Update the header and body for the current sample."""
        if not self.is_mounted:
            return
        header = self.query_one("#preview-header", Static)
        body = self.query_one("#preview-body", Static)
        if not self._samples:
            header.update("No samples to preview")
            body.update("")
            return
        text, tokens = self._samples[self._index]
        header.update(f"Sample {self._index + 1}/{len(self._samples)}  ·  {tokens:,} tokens")
        body.update(text)
