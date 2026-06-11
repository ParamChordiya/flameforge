"""Command-line entry point: ``flameforge`` and ``python -m flameforge``.

The CLI is intentionally thin. Its job is to parse a few global options, detect
the device (optionally with a user memory cap), and launch the Textual TUI where
the real interaction happens.
"""

from __future__ import annotations

import typer

from flameforge import __version__
from flameforge.constants import APP_NAME

app = typer.Typer(
    name="flameforge",
    help=f"{APP_NAME} — fine-tune any LLM from a beautiful terminal UI.",
    add_completion=False,
    no_args_is_help=False,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is passed."""
    if value:
        typer.echo(f"{APP_NAME} {__version__}")
        raise typer.Exit()


@app.command()
def run(
    max_memory_gb: float | None = typer.Option(
        None,
        "--max-memory-gb",
        help="Hard cap on the training memory budget, in GB (risky if set high).",
        min=0.5,
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Pre-select a HuggingFace model id or local path.",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Launch the FlameForge TUI."""
    # Imported lazily so ``--version``/``--help`` are instant and never touch the
    # (heavier) Textual + detection import chain.
    from flameforge.app import FlameForgeApp
    from flameforge.device.detector import detect_device

    device = detect_device(user_cap_gb=max_memory_gb)
    tui = FlameForgeApp(device=device)
    if model:
        tui.session.model_id = model
    tui.run()


def main() -> None:
    """Console-script entry point used by the ``flameforge`` command."""
    app()


if __name__ == "__main__":
    main()
