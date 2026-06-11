"""Method selection screen: pick LoRA, QLoRA, full fine-tuning, or DoRA.

Each method is shown as a card with a one-line description and an estimated peak
memory figure for the selected model. Methods that won't fit the device budget
are disabled with an explanation, and the cheapest fitting (and registry
recommended) method is highlighted.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Static

from flameforge.constants import FineTuningMethod
from flameforge.device.memory import estimate_model_memory_gb
from flameforge.utils.logging import get_logger

_log = get_logger("ui.method_select")


@dataclass(frozen=True)
class _MethodCard:
    """Static descriptive copy for one fine-tuning method."""

    method: FineTuningMethod
    title: str
    description: str


_CARDS = [
    _MethodCard(
        FineTuningMethod.QLORA,
        "QLoRA — 4-bit + LoRA adapters",
        "Lowest memory. Loads the base model in 4-bit and trains small adapters. Best default for large models.",
    ),
    _MethodCard(
        FineTuningMethod.LORA,
        "LoRA — low-rank adapters",
        "Trains small adapters on the 16-bit base model. Fast, high quality, modest memory.",
    ),
    _MethodCard(
        FineTuningMethod.DORA,
        "DoRA — weight-decomposed LoRA",
        "A LoRA refinement that often matches full fine-tuning quality at similar adapter cost.",
    ),
    _MethodCard(
        FineTuningMethod.FULL,
        "Full fine-tuning",
        "Updates every weight. Highest quality and highest memory — only for small models or big GPUs.",
    ),
]


class MethodSelectScreen(Screen[None]):
    """Lets the user choose a fine-tuning method for the selected model."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Build the method cards."""
        yield Header(show_clock=False)
        session = self.app.session  # type: ignore[attr-defined]
        budget = session.device.memory_budget_gb
        param_count = session.model_param_count
        recommended = self._recommended_method()

        with VerticalScroll(id="method-root"):
            model_name = session.model_info.display_name if session.model_info else session.model_id
            yield Label(f"Step 2 — Choose a method for {model_name}", classes="title")
            if param_count is None:
                yield Static(
                    "Memory estimates are unavailable for this custom model, so all methods are enabled. "
                    "QLoRA is the safest starting point.",
                    classes="hint",
                )
            else:
                yield Static(f"Your memory budget is {budget:.1f} GB.", classes="hint")

            for card in _CARDS:
                yield self._build_card(card, param_count, budget, recommended)
        yield Footer()

    def _build_card(
        self,
        card: _MethodCard,
        param_count: int | None,
        budget: float,
        recommended: FineTuningMethod,
    ) -> Vertical:
        """Construct a single method card widget."""
        fits = True
        if param_count is not None:
            estimate = estimate_model_memory_gb(param_count, card.method)
            fits = estimate <= budget
            mem_line = f"Estimated memory: {estimate:.1f} GB  ·  {'fits ✓' if fits else 'too large ✗'}"
        else:
            mem_line = "Estimated memory: unknown (custom model)"

        is_recommended = card.method == recommended and fits
        btn = Button(
            "Select" if fits else "Won't fit in budget",
            id=f"select-{card.method.value}",
            variant="success" if is_recommended else "primary",
        )
        btn.disabled = not fits

        container = Vertical(
            Static(card.description),
            Static(mem_line, classes="ok" if fits else "error"),
            btn,
            classes="method-card" + (" recommended" if is_recommended else ""),
        )
        container.border_title = card.title + ("  ★ Recommended" if is_recommended else "")
        return container

    def _recommended_method(self) -> FineTuningMethod:
        """Pick the recommended method: cheapest registry-recommended that fits."""
        session = self.app.session  # type: ignore[attr-defined]
        budget = session.device.memory_budget_gb
        param_count = session.model_param_count
        info = session.model_info

        if param_count is None:
            return FineTuningMethod.QLORA

        candidates = info.recommended_methods if info else ["qlora", "lora", "dora", "full"]
        fitting = [FineTuningMethod(m) for m in candidates if estimate_model_memory_gb(param_count, m) <= budget]
        if fitting:
            # Prefer the lowest-memory option among those that fit.
            return min(fitting, key=lambda m: estimate_model_memory_gb(param_count, m))
        return FineTuningMethod.QLORA

    def action_go_back(self) -> None:
        """Return to model selection."""
        self.app.go_back()  # type: ignore[attr-defined]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Select the chosen method and advance to data loading."""
        if event.button.id and event.button.id.startswith("select-"):
            method = FineTuningMethod(event.button.id.removeprefix("select-"))
            self.app.session.method = method  # type: ignore[attr-defined]
            _log.info("Method selected: %s", method.value)
            self.app.advance_to("data_load")  # type: ignore[attr-defined]
