"""HuggingFace authentication modal — frictionless, step-by-step token entry.

Shown when a gated model is selected and no token is available. It explains
exactly what to do (create an account, accept the license, make a token),
validates the entered token against the Hub in a worker thread, and optionally
saves it to the standard cache. The user is never left staring at a raw 401.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message as TextualMessage
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Static

from flameforge.utils.errors import AuthenticationError
from flameforge.utils.hf_utils import save_hf_token, validate_token
from flameforge.utils.logging import get_logger

_log = get_logger("ui.auth")


class AuthScreen(ModalScreen[bool]):
    """Modal that collects and validates a HuggingFace token for a gated model.

    Dismisses with ``True`` when a valid token is provided (and the user may
    proceed), or ``False`` if they choose a different model instead.

    Args:
        model_id: The gated model the user is trying to access.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    class Validated(TextualMessage):
        """Worker result: token validated successfully for ``username``."""

        def __init__(self, username: str, save: bool, token: str) -> None:
            self.username = username
            self.save = save
            self.token = token
            super().__init__()

    class Rejected(TextualMessage):
        """Worker result: token validation failed with ``error``."""

        def __init__(self, error: AuthenticationError) -> None:
            self.error = error
            super().__init__()

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._model_id = model_id

    def compose(self) -> ComposeResult:
        """Build the auth modal."""
        with Vertical(id="auth-card"):
            yield Label("🔑 HuggingFace Authentication Required", classes="title")
            yield Static(
                f"The model '{self._model_id}' requires a free HuggingFace account and an access token.",
            )
            yield Static(
                "Steps:\n"
                "  1. Create an account: https://huggingface.co/join\n"
                f"  2. Accept the license: https://huggingface.co/{self._model_id}\n"
                "  3. Create a token (Read scope): https://huggingface.co/settings/tokens\n"
                "  4. Paste the token below (or run: huggingface-cli login).",
                classes="hint",
            )
            yield Input(placeholder="hf_xxxxxxxxxxxxxxxxxxxx", password=True, id="auth-token")
            yield Checkbox("Save token for future use (~/.cache/huggingface/token)", id="auth-save", value=True)
            yield Static("", id="auth-status")
            with Horizontal(id="auth-buttons"):
                yield Button("Submit", id="auth-submit", variant="primary")
                yield Button("Choose Different Model", id="auth-cancel")

    def on_mount(self) -> None:
        """Focus the token field on open."""
        self.query_one("#auth-token", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit when the user presses Enter in the token field."""
        if event.input.id == "auth-token":
            self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Submit / Choose Different Model."""
        if event.button.id == "auth-submit":
            self._submit()
        elif event.button.id == "auth-cancel":
            self.action_cancel()

    def action_cancel(self) -> None:
        """Dismiss the modal, signalling the user wants a different model."""
        self.dismiss(False)

    def _submit(self) -> None:
        """Validate the entered token in a worker thread."""
        token = self.query_one("#auth-token", Input).value.strip()
        status = self.query_one("#auth-status", Static)
        if not token:
            status.set_class(True, "warn")
            status.update("Please paste a token, or press Escape to choose another model.")
            return
        save = self.query_one("#auth-save", Checkbox).value
        status.set_class(False, "warn", "error")
        status.set_class(True, "hint")
        status.update("Validating token…")

        def work() -> None:
            try:
                username = validate_token(token)
                self.post_message(self.Validated(username, bool(save), token))
            except AuthenticationError as exc:
                self.post_message(self.Rejected(exc))

        self.run_worker(work, thread=True, exclusive=True)

    def on_auth_screen_validated(self, message: Validated) -> None:
        """Save the token if requested and dismiss successfully."""
        if message.save:
            try:
                save_hf_token(message.token)
            except AuthenticationError as exc:
                # Saving is best-effort; a valid token still lets the user proceed.
                _log.warning("token save failed: %s", exc.message)
        self.notify(f"Authenticated as {message.username}.", title="Signed in")
        self.dismiss(True)

    def on_auth_screen_rejected(self, message: Rejected) -> None:
        """Show why the token was rejected."""
        status = self.query_one("#auth-status", Static)
        status.set_class(False, "hint", "warn")
        status.set_class(True, "error")
        status.update(message.error.format_plain())
