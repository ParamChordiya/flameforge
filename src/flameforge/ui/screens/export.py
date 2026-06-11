"""Export screen: summarise the run and produce the chosen artifacts.

Offers three artifact types — the LoRA adapter (smallest, already written during
training), a merged standalone model, and a GGUF file for llama.cpp/Ollama. Merge
and GGUF conversion run on a worker thread and report success (with copy-paste
usage instructions) or a friendly error. The screen also surfaces a clear note
when the preceding run was simulated and therefore produced no real weights.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message as TextualMessage
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Select, Static

from flameforge.constants import GGUF_QUANT_LEVELS, Backend
from flameforge.export.convert import convert_to_gguf
from flameforge.export.merge import merge_adapter
from flameforge.training.base import TrainingResult
from flameforge.ui.widgets.stage_bar import StageBar
from flameforge.utils.errors import FlameForgeError
from flameforge.utils.logging import get_logger

_log = get_logger("ui.export")


class ExportScreen(Screen[None]):
    """Final screen: training summary and export options."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+q", "quit", "Quit"),
    ]

    class ExportDone(TextualMessage):
        """Worker result: an export finished, with usage instructions."""

        def __init__(self, title: str, path: Path, usage: str) -> None:
            self.title = title
            self.path = path
            self.usage = usage
            super().__init__()

    class ExportFailed(TextualMessage):
        """Worker result: an export failed."""

        def __init__(self, error: FlameForgeError) -> None:
            self.error = error
            super().__init__()

    def compose(self) -> ComposeResult:
        """Build the export layout."""
        yield Header(show_clock=False)
        session = self.app.session  # type: ignore[attr-defined]
        result: TrainingResult | None = session.training_result
        with VerticalScroll(id="export-root"):
            yield StageBar(current=6)
            yield Label("Step 6 — Export your model", classes="title")
            yield Static(self._summary_text(result), id="export-summary", classes="panel")
            if result is not None and result.simulated:
                yield Static(
                    "⚠ The previous run was simulated, so no real weights were written. "
                    "Install a backend and train for real to produce exportable artifacts.",
                    classes="warn",
                )
            yield Label("Choose what to export:", classes="title")
            with Horizontal(id="export-buttons"):
                yield Button("Save Adapter", id="export-adapter", variant="success")
                yield Button("Merge Full Model", id="export-merged", variant="primary")
                yield Button("Convert to GGUF", id="export-gguf")
            yield Select(
                [(q, q) for q in GGUF_QUANT_LEVELS],
                value="Q4_K_M",
                id="gguf-quant",
                allow_blank=False,
            )
            yield Static("", id="export-status")
            yield Static("", id="export-usage", classes="panel")
            with Horizontal(id="export-finish"):
                yield Button("New Training Run", id="new-run")
                yield Button("Exit", id="exit-app", variant="error")
        yield Footer()

    def _summary_text(self, result: TrainingResult | None) -> str:
        """Build the training-summary block."""
        session = self.app.session  # type: ignore[attr-defined]
        model = session.model_info.display_name if session.model_info else session.model_id
        if result is None:
            return f"Model: {model}\n(No training summary available.)"
        mins = result.elapsed_sec / 60
        return (
            f"Model:        {model}\n"
            f"Method:       {session.method.value.upper() if session.method else '—'}\n"
            f"Final loss:   {result.final_loss:.4f}   (best {result.best_loss:.4f})\n"
            f"Steps:        {result.steps_completed:,}{'  (stopped early)' if result.stopped_early else ''}\n"
            f"Time:         {mins:.1f} min\n"
            f"Tokens:       ~{result.total_tokens:,}\n"
            f"Output dir:   {result.output_dir}"
        )

    # -- Navigation ------------------------------------------------------

    def action_go_back(self) -> None:
        """Return to the training screen."""
        self.app.go_back()  # type: ignore[attr-defined]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch export and finish actions."""
        match event.button.id:
            case "export-adapter":
                self._export_adapter()
            case "export-merged":
                self._run_export("merged")
            case "export-gguf":
                self._run_export("gguf")
            case "new-run":
                self.app.advance_to("model_select")  # type: ignore[attr-defined]
            case "exit-app":
                self.app.exit()

    # -- Export actions --------------------------------------------------

    def _export_adapter(self) -> None:
        """Report the adapter location and usage (it is written during training)."""
        session = self.app.session  # type: ignore[attr-defined]
        path = session.output_dir
        usage = (
            "Load the adapter on top of the base model:\n\n"
            "  from peft import PeftModel\n"
            "  from transformers import AutoModelForCausalLM\n"
            f"  base = AutoModelForCausalLM.from_pretrained('{session.model_id}')\n"
            f"  model = PeftModel.from_pretrained(base, '{path}')"
        )
        self._show_result("Adapter ready", Path(path), usage)

    def _run_export(self, kind: str) -> None:
        """Run a merge or GGUF export on a worker thread."""
        session = self.app.session  # type: ignore[attr-defined]
        status = self.query_one("#export-status", Static)
        status.set_class(False, "error")
        status.update("Exporting… this can take a few minutes for large models.")
        model_id = session.model_id
        adapter_path = Path(session.output_dir)
        backend = session.device.backend
        merged_dir = adapter_path / "merged"
        quant = self.query_one("#gguf-quant", Select).value

        def work() -> None:
            try:
                if kind == "merged":
                    out = merge_adapter(model_id, adapter_path, merged_dir, backend)
                    usage = _merged_usage(out, backend)
                    self.post_message(self.ExportDone("Merged model saved", out, usage))
                else:
                    if not merged_dir.exists():
                        merge_adapter(model_id, adapter_path, merged_dir, backend)
                    gguf_path = adapter_path / f"model-{quant}.gguf"
                    out = convert_to_gguf(merged_dir, gguf_path, quantization=str(quant))
                    usage = _gguf_usage(out)
                    self.post_message(self.ExportDone("GGUF saved", out, usage))
            except FlameForgeError as exc:
                self.post_message(self.ExportFailed(exc))

        self.run_worker(work, thread=True, exclusive=True)

    def on_export_screen_export_done(self, message: ExportDone) -> None:
        """Show a successful export's location and usage."""
        self.query_one("#export-status", Static).update("")
        self._show_result(message.title, message.path, message.usage)

    def on_export_screen_export_failed(self, message: ExportFailed) -> None:
        """Show a friendly error when an export fails."""
        status = self.query_one("#export-status", Static)
        status.set_class(True, "error")
        status.update(message.error.format_plain())
        self.query_one("#export-usage", Static).update("")

    def _show_result(self, title: str, path: Path, usage: str) -> None:
        """Render an export result block."""
        self.query_one("#export-status", Static).update(f"✓ {title} → {path}")
        self.query_one("#export-usage", Static).update(usage)


def _merged_usage(path: Path, backend: Backend) -> str:
    """Usage instructions for a merged model."""
    if backend == Backend.MLX:
        return f"Run it with mlx-lm:\n\n  mlx_lm.generate --model {path} --prompt 'Hello'"
    return (
        "Load it like any HuggingFace model:\n\n"
        "  from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f"  model = AutoModelForCausalLM.from_pretrained('{path}')\n"
        f"  tok = AutoTokenizer.from_pretrained('{path}')"
    )


def _gguf_usage(path: Path) -> str:
    """Usage instructions for a GGUF file."""
    return (
        "Use it with llama.cpp or Ollama:\n\n"
        f"  llama-cli -m {path} -p 'Hello'\n\n"
        "  # or with Ollama, create a Modelfile:\n"
        f"  FROM {path}"
    )
