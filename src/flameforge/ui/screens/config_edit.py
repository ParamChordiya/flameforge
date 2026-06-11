"""Configuration screen: auto-tuned hyperparameters the user can review/edit.

On entry the screen runs :func:`flameforge.training.hyperparams.auto_tune` and
pre-fills a form with the suggested values, explaining each adjustment inline.
Common fields are always visible; the rest live under an "Advanced" collapsible.
Every value is validated through :class:`~flameforge.config.TrainingConfig` before
training can start, so an invalid number is caught here, not mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Grid, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Collapsible, Footer, Header, Input, Label, Static

from flameforge.config import TrainingConfig
from flameforge.training.hyperparams import auto_tune
from flameforge.utils.errors import ConfigurationError
from flameforge.utils.logging import get_logger

_log = get_logger("ui.config_edit")


@dataclass(frozen=True)
class _Field:
    """Describes one editable numeric/text field in the form."""

    name: str
    label: str
    kind: str  # "int", "float", or "str"
    advanced: bool


_FIELDS = [
    _Field("num_epochs", "Epochs", "int", False),
    _Field("learning_rate", "Learning rate", "float", False),
    _Field("max_seq_length", "Max sequence length", "int", False),
    _Field("per_device_batch_size", "Micro-batch size", "int", False),
    _Field("gradient_accumulation", "Grad accumulation", "int", False),
    _Field("effective_batch_size", "Effective batch size", "int", False),
    _Field("lora_rank", "LoRA rank", "int", True),
    _Field("lora_alpha", "LoRA alpha", "int", True),
    _Field("lora_dropout", "LoRA dropout", "float", True),
    _Field("warmup_ratio", "Warmup ratio", "float", True),
    _Field("weight_decay", "Weight decay", "float", True),
    _Field("save_steps", "Save every N steps", "int", True),
    _Field("save_total_limit", "Keep N checkpoints", "int", True),
    _Field("seed", "Random seed", "int", True),
    _Field("output_dir", "Output directory", "str", True),
]


class ConfigEditScreen(Screen[None]):
    """Lets the user review and edit auto-tuned training hyperparameters."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Build the configuration form."""
        yield Header(show_clock=False)
        with VerticalScroll(id="config-root"):
            yield Label("Step 4 — Review training configuration", classes="title")
            yield Static(
                "These values were auto-tuned for your model, hardware, and dataset. "
                "Adjust anything you like, or just start.",
                classes="hint",
            )
            yield Static("", id="autotune-warnings", classes="panel")
            with Grid(id="config-grid"):
                for field in _FIELDS:
                    if not field.advanced:
                        yield Label(field.label)
                        yield Input(id=f"cfg-{field.name}")
            with Collapsible(title="Advanced settings", collapsed=True):
                with Grid(id="config-grid-advanced"):
                    for field in _FIELDS:
                        if field.advanced:
                            yield Label(field.label)
                            yield Input(id=f"cfg-{field.name}")
                yield Checkbox("Gradient checkpointing (saves memory)", id="cfg-gradient_checkpointing")
                yield Checkbox("Use bf16 precision", id="cfg-bf16")
            yield Static("", id="config-error", classes="error")
            with Vertical(id="config-actions"):
                yield Button("Start Training ▶", id="start-training", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        """Auto-tune and populate the form."""
        session = self.app.session  # type: ignore[attr-defined]
        if session.method is None:
            return
        result = auto_tune(
            session.training,
            session.method,
            session.device,
            dataset_size=session.data.num_examples if session.data else 0,
            model_info=session.model_info,
        )
        session.training = result.config
        self._populate(result.config)
        warnings = self.query_one("#autotune-warnings", Static)
        if result.warnings:
            warnings.update("Auto-tuning notes:\n" + "\n".join(f"  • {w}" for w in result.warnings))
        else:
            warnings.update("Defaults look good for this setup — no adjustments needed.")

    def _populate(self, config: TrainingConfig) -> None:
        """Fill every form input from a config object."""
        for field in _FIELDS:
            self.query_one(f"#cfg-{field.name}", Input).value = str(getattr(config, field.name))
        self.query_one("#cfg-gradient_checkpointing", Checkbox).value = config.gradient_checkpointing
        self.query_one("#cfg-bf16", Checkbox).value = config.bf16

    def action_go_back(self) -> None:
        """Return to the data step."""
        self.app.go_back()  # type: ignore[attr-defined]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Validate and start training when requested."""
        if event.button.id == "start-training":
            self._start()

    def _collect(self) -> dict[str, object]:
        """Read every input into a typed dict for TrainingConfig construction.

        Raises:
            ConfigurationError: If a numeric field contains an unparseable value.
        """
        values: dict[str, object] = {}
        for field in _FIELDS:
            raw = self.query_one(f"#cfg-{field.name}", Input).value.strip()
            if field.kind == "int":
                values[field.name] = _parse_number(raw, field.label, integer=True)
            elif field.kind == "float":
                values[field.name] = _parse_number(raw, field.label, integer=False)
            else:
                values[field.name] = raw
        values["gradient_checkpointing"] = self.query_one("#cfg-gradient_checkpointing", Checkbox).value
        bf16 = self.query_one("#cfg-bf16", Checkbox).value
        values["bf16"] = bf16
        # bf16 and fp16 are mutually exclusive; preserve an auto-tuned fp16 only
        # when the user has turned bf16 off.
        values["fp16"] = (not bf16) and self.app.session.training.fp16  # type: ignore[attr-defined]
        return values

    def _start(self) -> None:
        """Build a validated TrainingConfig and advance to training."""
        error = self.query_one("#config-error", Static)
        try:
            config = TrainingConfig.model_validate(self._collect())
        except ConfigurationError as exc:
            error.update(f"✗ {exc.message}")
            return
        except Exception as exc:  # pydantic ValidationError → inline message
            error.update(f"✗ Invalid configuration: {exc}")
            return
        error.update("")
        self.app.session.training = config  # type: ignore[attr-defined]
        _log.info("Training config confirmed; starting run")
        self.app.advance_to("training")  # type: ignore[attr-defined]


def _parse_number(raw: str, label: str, integer: bool) -> int | float:
    """Parse a numeric form value, raising a friendly error on failure."""
    try:
        return int(raw) if integer else float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            message=f"'{label}' must be {'a whole number' if integer else 'a number'} (got '{raw}').",
        ) from exc
