"""Data loading screen: pick a file, auto-detect, validate, and preview.

The user types or pastes a path; FlameForge detects the format, loads and
normalizes the data, validates it, and shows both a validation report and a
faithful formatted preview. Loading runs in a worker thread so the UI never
freezes on a large file, and any failure is rendered as a friendly error panel
rather than a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from flameforge.config import DataConfig
from flameforge.constants import DataFormat
from flameforge.data.detector import detect_format
from flameforge.data.formatter import Formatter
from flameforge.data.loader import Dataset, load_dataset
from flameforge.data.templates import resolve_template_key
from flameforge.data.validator import IssueLevel, ValidationReport, validate_dataset
from flameforge.ui.widgets.data_preview import DataPreview
from flameforge.utils.errors import FlameForgeError
from flameforge.utils.logging import get_logger

_log = get_logger("ui.data_load")
_PREVIEW_LIMIT = 5

_LEVEL_GLYPH = {IssueLevel.OK: "✓", IssueLevel.WARNING: "⚠", IssueLevel.ERROR: "✗"}


@dataclass
class _LoadResult:
    """Everything produced by a successful load, ready to render."""

    dataset: Dataset
    report: ValidationReport
    samples: list[tuple[str, int]]
    template_key: str


class DataLoadScreen(Screen[None]):
    """Screen for selecting, validating, and previewing a dataset."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+q", "quit", "Quit"),
    ]

    class LoadSucceeded(TextualMessage):
        """Posted from the worker thread when loading + validation succeed."""

        def __init__(self, result: _LoadResult) -> None:
            self.result = result
            super().__init__()

    class LoadFailed(TextualMessage):
        """Posted from the worker thread when loading fails."""

        def __init__(self, error: FlameForgeError) -> None:
            self.error = error
            super().__init__()

    def compose(self) -> ComposeResult:
        """Build the data-loading layout."""
        yield Header(show_clock=False)
        with VerticalScroll(id="data-root"):
            yield Label("Step 3 — Load your training data", classes="title")
            yield Static(
                "Enter a path to a .jsonl, .json, .csv, .tsv, or .txt file. "
                "Try one of the bundled examples in ./examples to start.",
                classes="hint",
            )
            with Horizontal(id="data-input-row"):
                yield Input(
                    placeholder="examples/sample_alpaca.jsonl",
                    id="data-path",
                )
                yield Button("Load", id="load-data", variant="primary")
            yield Static("", id="load-status")
            with Vertical(id="data-results"):
                yield Static("", id="format-badge", classes="title")
                yield Static("", id="validation-report", classes="panel")
                yield DataPreview()
                with Horizontal(id="data-actions"):
                    yield Button("Confirm & Continue ▶", id="confirm-data", variant="success")
                    yield Button("Choose Different File", id="reset-data")
        yield Footer()

    def on_mount(self) -> None:
        """Hide the results area until data has been loaded."""
        self.query_one("#data-results").display = False

    # -- Actions ---------------------------------------------------------

    def action_go_back(self) -> None:
        """Return to the previous step."""
        self.app.go_back()  # type: ignore[attr-defined]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Trigger a load when the user presses Enter in the path field."""
        if event.input.id == "data-path":
            self._begin_load()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch button presses for load/confirm/reset."""
        if event.button.id == "load-data":
            self._begin_load()
        elif event.button.id == "confirm-data":
            self._confirm()
        elif event.button.id == "reset-data":
            self._reset()

    # -- Loading flow ----------------------------------------------------

    def _begin_load(self) -> None:
        """Validate the path field and kick off the background load."""
        raw = self.query_one("#data-path", Input).value.strip()
        status = self.query_one("#load-status", Static)
        if not raw:
            status.update("Please enter a file path.")
            status.set_class(True, "warn")
            return
        path = Path(raw).expanduser()
        if not path.is_file():
            status.update(f"✗ File not found: {path}")
            status.set_class(True, "error")
            return
        status.set_class(False, "error", "warn")
        status.update("Loading and validating…")
        self.query_one("#data-results").display = False
        self._load_worker(path)

    def _load_worker(self, path: Path) -> None:
        """Run loading + validation in a thread and post the outcome back."""

        def work() -> None:
            try:
                fmt = detect_format(path)
                dataset = load_dataset(path, data_format=fmt)
                family = self._family()
                template_key = resolve_template_key(family=family)
                formatter = Formatter(template_key=template_key)
                report = validate_dataset(dataset, formatter=formatter)
                samples = [formatter.format_with_token_count(ex) for ex in dataset.examples[:_PREVIEW_LIMIT]]
                result = _LoadResult(dataset, report, samples, template_key)
                self.post_message(self.LoadSucceeded(result))
            except FlameForgeError as exc:
                _log.warning("data load failed: %s", exc.message)
                self.post_message(self.LoadFailed(exc))
            except Exception as exc:  # pragma: no cover - unexpected, wrapped for safety
                _log.exception("unexpected data load error")
                self.post_message(self.LoadFailed(FlameForgeError(f"Unexpected error: {exc}")))

        self.run_worker(work, thread=True, exclusive=True)

    def _family(self) -> str | None:
        """Return the selected model's family, if a model has been chosen."""
        info = self.app.session.model_info  # type: ignore[attr-defined]
        return info.family if info else None

    def on_data_load_screen_load_succeeded(self, message: LoadSucceeded) -> None:
        """Render results when a background load completes successfully."""
        self._result = message.result
        status = self.query_one("#load-status", Static)
        status.update("")
        self._render_results(message.result)

    def on_data_load_screen_load_failed(self, message: LoadFailed) -> None:
        """Show a friendly error panel when a background load fails."""
        status = self.query_one("#load-status", Static)
        status.set_class(True, "error")
        status.update(message.error.format_plain())

    def _render_results(self, result: _LoadResult) -> None:
        """Populate the format badge, validation report, and preview."""
        results = self.query_one("#data-results")
        results.display = True

        fmt_name = _format_label(result.dataset.data_format)
        self.query_one("#format-badge", Static).update(
            f"Detected format: {fmt_name}  ·  template: {result.template_key}"
        )

        report = result.report
        lines = [f"  {_LEVEL_GLYPH[i.level]} {i.message}" for i in report.issues]
        lines.append("")
        lines.append(f"  Train / Eval split: {report.train_count:,} / {report.eval_count:,}")
        lines.append(f"  Avg tokens/example: {report.avg_tokens:.0f}   Max: {report.max_tokens:,}")
        lines.append(f"  Total training tokens: {report.total_tokens:,}")
        self.query_one("#validation-report", Static).update("\n".join(lines))

        self.query_one(DataPreview).set_samples(result.samples)

        confirm = self.query_one("#confirm-data", Button)
        confirm.disabled = not report.is_trainable

    def _confirm(self) -> None:
        """Persist the loaded dataset to the session and advance."""
        result = getattr(self, "_result", None)
        if result is None or not result.report.is_trainable:
            return
        session = self.app.session  # type: ignore[attr-defined]
        session.data = DataConfig(
            path=result.dataset.source_path,
            data_format=result.dataset.data_format,
            num_examples=result.report.valid_examples,
        )
        session.chat_template = result.template_key
        self.app.advance_to("config_edit")  # type: ignore[attr-defined]

    def _reset(self) -> None:
        """Clear results so the user can choose a different file."""
        self.query_one("#data-results").display = False
        self.query_one("#data-path", Input).value = ""
        self.query_one("#data-path", Input).focus()


def _format_label(fmt: DataFormat) -> str:
    """Return a human-friendly label for a detected data format."""
    return {
        DataFormat.ALPACA: "Alpaca (instruction / input / output)",
        DataFormat.CONVERSATIONAL: "Conversational (chat messages)",
        DataFormat.TEXT: "Raw text (language modeling)",
        DataFormat.CSV: "CSV / TSV (column-mapped)",
    }[fmt]
