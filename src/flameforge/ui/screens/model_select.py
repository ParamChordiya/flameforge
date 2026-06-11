"""Model selection screen: browse popular models, search the Hub, or go local.

Three tabs cover every way a user might pick a model. The "Popular" tab shows a
fit-checked table driven by the registry; "Search" queries the Hub live in a
worker thread; "Local Path" accepts a directory. Selecting a gated model with no
token on hand opens the auth modal before proceeding.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message as TextualMessage
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from flameforge.models.registry import POPULAR_MODELS, ModelInfo, get_model
from flameforge.ui.screens.auth import AuthScreen
from flameforge.utils.hf_utils import HubModelResult, find_hf_token, is_local_model_path, search_hub_models
from flameforge.utils.logging import get_logger

_log = get_logger("ui.model_select")


class ModelSelectScreen(Screen[None]):
    """Lets the user choose a model to fine-tune."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+q", "quit", "Quit"),
    ]

    class SearchDone(TextualMessage):
        """Worker result carrying Hub search results (or an error message)."""

        def __init__(self, results: list[HubModelResult], error: str | None) -> None:
            self.results = results
            self.error = error
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self._popular_keys: dict[str, ModelInfo] = {}
        self._search_keys: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        """Build the tabbed model browser."""
        yield Header(show_clock=False)
        budget = self.app.session.device.memory_budget_gb  # type: ignore[attr-defined]
        with Vertical(id="model-root"):
            yield Label("Step 1 — Choose a model", classes="title")
            yield Static(
                f"A ✓ means the model fits your {budget:.1f} GB budget with its cheapest method.",
                classes="hint",
            )
            with TabbedContent(initial="tab-popular"):
                with TabPane("Popular Models", id="tab-popular"):
                    yield DataTable(id="popular-table", cursor_type="row")
                with TabPane("Search HuggingFace", id="tab-search"):
                    with Horizontal(id="search-row"):
                        yield Input(placeholder="e.g. llama, mistral, code", id="search-input")
                        yield Button("Search", id="search-btn", variant="primary")
                    yield Static("", id="search-status", classes="hint")
                    yield DataTable(id="search-table", cursor_type="row")
                with TabPane("Local Path", id="tab-local"):
                    yield Static("Point FlameForge at a local model directory.", classes="hint")
                    with Horizontal(id="local-row"):
                        yield Input(placeholder="/path/to/model", id="local-input")
                        yield Button("Use This Model", id="local-btn", variant="primary")
                    yield Static("", id="local-status")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the popular-models table."""
        budget = self.app.session.device.memory_budget_gb  # type: ignore[attr-defined]
        table = self.query_one("#popular-table", DataTable)
        table.add_columns("Model", "Size", "Context", "License", "Fits")
        for model in POPULAR_MODELS:
            fits = "✓" if model.fits_in_budget(budget) else "✗"
            auth = " 🔒" if model.requires_auth else ""
            key = table.add_row(
                f"{model.display_name}{auth}",
                model.param_count_str,
                f"{model.context_length // 1024}k" if model.context_length >= 1024 else str(model.context_length),
                model.license,
                fits,
            )
            self._popular_keys[str(key.value)] = model

        search_table = self.query_one("#search-table", DataTable)
        search_table.add_columns("Model ID", "Downloads", "Likes", "Gated")

    # -- Navigation ------------------------------------------------------

    def action_go_back(self) -> None:
        """Return to the welcome screen."""
        self.app.go_back()  # type: ignore[attr-defined]

    # -- Popular tab -----------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle a row selection in either the popular or search table."""
        row_key = str(event.row_key.value)
        if event.data_table.id == "popular-table":
            model = self._popular_keys.get(row_key)
            if model is not None:
                self._choose_registry_model(model)
        elif event.data_table.id == "search-table":
            model_id = self._search_keys.get(row_key)
            if model_id is not None:
                self._choose_custom(model_id)

    # -- Search tab ------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Trigger search or local selection on Enter."""
        if event.input.id == "search-input":
            self._run_search()
        elif event.input.id == "local-input":
            self._choose_local()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Search / Use This Model buttons."""
        if event.button.id == "search-btn":
            self._run_search()
        elif event.button.id == "local-btn":
            self._choose_local()

    def _run_search(self) -> None:
        """Query the Hub in a worker thread."""
        query = self.query_one("#search-input", Input).value.strip()
        status = self.query_one("#search-status", Static)
        if not query:
            status.update("Enter a search term.")
            return
        status.update(f"Searching the Hub for '{query}'…")

        def work() -> None:
            try:
                results = search_hub_models(query, limit=25)
                self.post_message(self.SearchDone(results, None))
            except Exception as exc:  # search wraps its own errors; be defensive
                self.post_message(self.SearchDone([], str(exc)))

        self.run_worker(work, thread=True, exclusive=True)

    def on_model_select_screen_search_done(self, message: SearchDone) -> None:
        """Render Hub search results."""
        status = self.query_one("#search-status", Static)
        table = self.query_one("#search-table", DataTable)
        table.clear()
        self._search_keys.clear()
        if message.error:
            status.update(f"✗ {message.error}")
            return
        if not message.results:
            status.update("No models found. Try a different term, or use a local path.")
            return
        status.update(f"{len(message.results)} results — select one to use it.")
        for result in message.results:
            key = table.add_row(
                result.model_id,
                f"{result.downloads:,}",
                f"{result.likes:,}",
                "🔒" if result.gated else "",
            )
            self._search_keys[str(key.value)] = result.model_id

    # -- Local tab -------------------------------------------------------

    def _choose_local(self) -> None:
        """Validate and select a local model directory."""
        raw = self.query_one("#local-input", Input).value.strip()
        status = self.query_one("#local-status", Static)
        if not raw:
            status.set_class(True, "warn")
            status.update("Enter a path to a local model directory.")
            return
        if not is_local_model_path(raw):
            status.set_class(True, "error")
            status.update(f"✗ Not a directory: {raw}")
            return
        self._choose_custom(raw, local=True)

    # -- Selection logic -------------------------------------------------

    def _choose_registry_model(self, model: ModelInfo) -> None:
        """Select a known registry model, handling auth if it is gated."""
        session = self.app.session  # type: ignore[attr-defined]
        session.model_id = model.hf_id
        session.model_info = model
        session.reset_downstream_of_model()
        if model.requires_auth and find_hf_token() is None:
            self.app.push_screen(AuthScreen(model.hf_id), self._after_auth)  # type: ignore[attr-defined]
        else:
            self.app.advance_to("method_select")  # type: ignore[attr-defined]

    def _choose_custom(self, model_id: str, local: bool = False) -> None:
        """Select a searched or local model id."""
        session = self.app.session  # type: ignore[attr-defined]
        session.model_id = model_id
        session.model_info = get_model(model_id)  # may be None for unknown ids
        session.reset_downstream_of_model()
        self.app.advance_to("method_select")  # type: ignore[attr-defined]

    def _after_auth(self, authenticated: bool | None) -> None:
        """Continue to method selection only if authentication succeeded."""
        if authenticated:
            self.app.advance_to("method_select")  # type: ignore[attr-defined]
        else:
            self.notify("Pick a different model, or sign in to use that one.", title="Not signed in")
