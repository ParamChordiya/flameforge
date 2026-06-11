"""The live training dashboard — the centrepiece of the FlameForge experience.

Training runs on a worker thread; the screen receives metric snapshots as
messages and refreshes the loss chart, metrics panel, memory bar, and progress
on a throttled timer so the UI stays smooth even at high logging rates. The user
can pause/resume, force a checkpoint, or stop early at any time. When the run
finishes it stores the result and offers to continue to export.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, ProgressBar, Static

from flameforge.data.formatter import Formatter
from flameforge.data.loader import load_dataset
from flameforge.data.templates import resolve_template_key
from flameforge.training.base import BaseTrainer, TrainingData, TrainingMetrics, build_trainer
from flameforge.ui.widgets.loss_chart import LossChart
from flameforge.ui.widgets.memory_bar import MemoryBar
from flameforge.ui.widgets.metrics_panel import MetricsPanel
from flameforge.ui.widgets.stage_bar import StageBar
from flameforge.utils.errors import FlameForgeError
from flameforge.utils.logging import get_logger

_log = get_logger("ui.training")
_REFRESH_INTERVAL = 0.4  # seconds between throttled UI refreshes


class TrainingScreen(Screen[None]):
    """Runs training and renders a live dashboard."""

    BINDINGS = [
        ("p", "toggle_pause", "Pause/Resume"),
        ("s", "stop", "Stop early"),
        ("c", "checkpoint", "Checkpoint"),
        ("ctrl+q", "quit", "Quit"),
    ]

    class MetricsUpdate(TextualMessage):
        """A new metrics snapshot from the training worker."""

        def __init__(self, metrics: TrainingMetrics) -> None:
            self.metrics = metrics
            super().__init__()

    class Finished(TextualMessage):
        """Training completed (or stopped) successfully."""

    class Failed(TextualMessage):
        """Training raised an error."""

        def __init__(self, error: FlameForgeError) -> None:
            self.error = error
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self._trainer: BaseTrainer | None = None
        self._latest: TrainingMetrics | None = None
        self._pending_losses: list[float] = []
        self._done = False

    def compose(self) -> ComposeResult:
        """Build the dashboard layout."""
        yield Header(show_clock=True)
        session = self.app.session  # type: ignore[attr-defined]
        with VerticalScroll(id="training-root"):
            yield StageBar(current=5)
            model = session.model_info.display_name if session.model_info else session.model_id
            method = session.method.value.upper() if session.method else "—"
            yield Static(
                f"Model: {model}   ·   Method: {method}   ·   Device: {session.device.device_name}",
                id="training-summary",
                classes="subtitle",
            )
            yield Static("", id="training-banner")
            yield ProgressBar(total=100, show_eta=False, id="training-progress")
            yield Static("Step 0", id="progress-label", classes="hint")
            yield Label("Loss", classes="title")
            yield LossChart()
            with Horizontal(id="dash-panels"):
                with Vertical(id="metrics-box", classes="panel"):
                    yield Label("Metrics", classes="title")
                    yield MetricsPanel()
                with Vertical(id="system-box", classes="panel"):
                    yield Label("System", classes="title")
                    yield MemoryBar()
                    yield Static("", id="system-extra")
            with Horizontal(id="training-controls"):
                yield Button("Pause", id="pause-btn", variant="warning")
                yield Button("Save Checkpoint", id="checkpoint-btn")
                yield Button("Stop Early", id="stop-btn", variant="error")
                yield Button("Continue to Export ▶", id="export-btn", variant="success", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        """Build the trainer and start training."""
        session = self.app.session  # type: ignore[attr-defined]
        run_config = session.build_run_config()
        num_examples = session.data.num_examples if session.data else 0
        self._trainer = build_trainer(run_config, session.device, num_examples)
        banner = self.query_one("#training-banner", Static)
        if self._trainer.simulated:
            banner.set_class(True, "warn")
            banner.update(
                "⚠ Simulation mode: no training backend is installed, so metrics are synthetic. "
                "Install flameforge[mlx] or flameforge[cuda] to train for real."
            )
        else:
            banner.update("Training in progress — you can pause, checkpoint, or stop at any time.")
        self.set_interval(_REFRESH_INTERVAL, self._refresh_ui)
        self._start_training()

    # -- Training worker -------------------------------------------------

    def _start_training(self) -> None:
        """Prepare data (if needed) and run training on a worker thread."""

        def work() -> None:
            trainer = self._trainer
            assert trainer is not None
            try:
                if not trainer.simulated:
                    trainer.set_data(self._prepare_data())
                for metrics in trainer.train():
                    self.post_message(self.MetricsUpdate(metrics))
                self.post_message(self.Finished())
            except FlameForgeError as exc:
                self.post_message(self.Failed(exc))
            except Exception as exc:  # pragma: no cover - defensive catch-all
                _log.exception("training worker crashed")
                self.post_message(self.Failed(FlameForgeError(f"Unexpected training error: {exc}")))

        self.run_worker(work, thread=True, exclusive=True)

    def _prepare_data(self) -> TrainingData:
        """Load, format, and split the dataset into train/eval text."""
        session = self.app.session  # type: ignore[attr-defined]
        data_cfg = session.data
        assert data_cfg is not None
        dataset = load_dataset(data_cfg.path, data_format=data_cfg.data_format, column_mapping=data_cfg.column_mapping)
        family = session.model_info.family if session.model_info else None
        template_key = resolve_template_key(template_key=session.chat_template, family=family)
        formatter = Formatter(template_key=template_key)
        texts = [formatter.format_example(ex) for ex in dataset.examples if formatter.format_example(ex).strip()]
        split = max(1, int(len(texts) * session.training.train_eval_split))
        train_texts = texts[:split]
        eval_texts = texts[split:] or texts[:1]
        return TrainingData(train_texts=train_texts, eval_texts=eval_texts)

    # -- Message handlers ------------------------------------------------

    def on_training_screen_metrics_update(self, message: MetricsUpdate) -> None:
        """Buffer a metrics snapshot for the next throttled refresh."""
        self._latest = message.metrics
        self._pending_losses.append(message.metrics.loss)

    def on_training_screen_finished(self, message: Finished) -> None:
        """Finalize the dashboard and enable export."""
        self._refresh_ui()  # flush any buffered points
        self._done = True
        session = self.app.session  # type: ignore[attr-defined]
        trainer = self._trainer
        if trainer is not None and trainer.result is not None:
            session.training_result = trainer.result
            session.output_dir = trainer.result.output_dir
        banner = self.query_one("#training-banner", Static)
        banner.set_class(False, "warn")
        banner.set_class(True, "ok")
        stopped = trainer.result.stopped_early if trainer and trainer.result else False
        banner.update("Training stopped early — your progress is saved." if stopped else "✓ Training complete!")
        self.query_one("#export-btn", Button).disabled = False
        for btn_id in ("pause-btn", "checkpoint-btn", "stop-btn"):
            self.query_one(f"#{btn_id}", Button).disabled = True

    def on_training_screen_failed(self, message: Failed) -> None:
        """Show a friendly error panel when training fails."""
        self._done = True
        banner = self.query_one("#training-banner", Static)
        banner.set_class(False, "warn", "ok")
        banner.set_class(True, "error")
        banner.update(message.error.format_plain())
        for btn_id in ("pause-btn", "checkpoint-btn", "stop-btn"):
            self.query_one(f"#{btn_id}", Button).disabled = True

    # -- Throttled refresh -----------------------------------------------

    def _refresh_ui(self) -> None:
        """Flush buffered metrics into the widgets (called on a timer)."""
        if self._pending_losses:
            self.query_one(LossChart).add_points(self._pending_losses)
            self._pending_losses = []
        m = self._latest
        if m is None:
            return
        self.query_one(MetricsPanel).update_metrics(m)
        self.query_one(MemoryBar).update_usage(m.mem_used_gb, m.mem_budget_gb)
        progress = self.query_one("#training-progress", ProgressBar)
        progress.update(total=100, progress=m.progress_fraction * 100)
        self.query_one("#progress-label", Static).update(
            f"Epoch {m.epoch:.2f}   ·   Step {m.step:,}/{m.total_steps:,}   ·   {m.progress_fraction * 100:.0f}%"
        )

    # -- Actions / controls ----------------------------------------------

    def action_toggle_pause(self) -> None:
        """Pause or resume training."""
        trainer = self._trainer
        if trainer is None or self._done:
            return
        button = self.query_one("#pause-btn", Button)
        if trainer.control.is_paused:
            trainer.control.resume()
            button.label = "Pause"
        else:
            trainer.control.pause()
            button.label = "Resume"

    def action_checkpoint(self) -> None:
        """Request an immediate checkpoint."""
        trainer = self._trainer
        if trainer is not None and not self._done:
            trainer.control.checkpoint_requested.set()
            self.notify("Checkpoint requested.", title="Checkpoint")

    def action_stop(self) -> None:
        """Stop training early after the next checkpoint."""
        trainer = self._trainer
        if trainer is not None and not self._done:
            trainer.control.stop_requested.set()
            self.notify("Stopping after the current step…", title="Stopping")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route control-button presses to their actions."""
        if event.button.id == "pause-btn":
            self.action_toggle_pause()
        elif event.button.id == "checkpoint-btn":
            self.action_checkpoint()
        elif event.button.id == "stop-btn":
            self.action_stop()
        elif event.button.id == "export-btn":
            self.app.advance_to("export")  # type: ignore[attr-defined]
