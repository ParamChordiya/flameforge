"""Custom exception classes with user-friendly, actionable messages.

Every error a user could plausibly hit during a FlameForge session should be
represented here as a :class:`FlameForgeError` subclass. The TUI never shows raw
tracebacks; instead it catches these exceptions and renders ``message``,
``suggestions``, and (optionally) ``details`` in a clean panel.

The golden rule: an error message must tell the user *what* went wrong, *why*,
and *what to do next*.
"""

from __future__ import annotations

from collections.abc import Iterable


class FlameForgeError(Exception):
    """Base class for all FlameForge errors.

    Args:
        message: A short, human-readable description of what went wrong.
        suggestions: Concrete, actionable steps the user can take to fix it.
        details: Optional technical detail (e.g. an upstream traceback string)
            useful for bug reports but not essential for the user to act.

    Attributes:
        message: The user-facing summary.
        suggestions: Ordered list of remediation steps.
        details: Optional technical context.
    """

    def __init__(
        self,
        message: str,
        suggestions: list[str] | None = None,
        details: str | None = None,
    ) -> None:
        self.message = message
        self.suggestions = suggestions or []
        self.details = details
        super().__init__(message)

    def format_plain(self) -> str:
        """Render the error as plain text suitable for logs or a non-TUI console.

        Returns:
            A multi-line string containing the message, any suggestions, and
            technical details if present.
        """
        lines = [f"✗ {self.message}"]
        if self.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            lines.extend(f"  • {s}" for s in self.suggestions)
        if self.details:
            lines.append("")
            lines.append("Technical details (for bug reports):")
            lines.append(f"  {self.details}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_plain()


class ModelNotFoundError(FlameForgeError):
    """Raised when a requested model cannot be found locally or on the Hub."""


class AuthenticationError(FlameForgeError):
    """Raised when a model requires HuggingFace auth that is missing or invalid."""


class OutOfMemoryError(FlameForgeError):
    """Raised when a model/method will not fit, or training exhausts the budget."""


class DataFormatError(FlameForgeError):
    """Raised when a data file cannot be parsed or its format is unsupported."""


class FormatDetectionError(DataFormatError):
    """Raised when a JSON/JSONL file's schema does not match any known format.

    Args:
        keys: The top-level keys found in the first record, used to build a
            helpful message showing the user what we saw.
        suggestions: Optional override for the default remediation steps.
    """

    def __init__(self, keys: Iterable[object], suggestions: list[str] | None = None) -> None:
        keys = list(keys)
        key_list = ", ".join(repr(k) for k in keys) if keys else "(none)"
        super().__init__(
            message=(f"Could not auto-detect the data format. The first record had keys: {key_list}."),
            suggestions=suggestions
            or [
                "Use Alpaca format: keys 'instruction', optional 'input', 'output'.",
                "Use chat format: a 'messages' or 'conversations' list of role/content turns.",
                "For raw text, save the file with a .txt extension instead.",
            ],
        )
        self.keys = keys


class UnsupportedFormatError(DataFormatError):
    """Raised when a file extension is not one FlameForge knows how to load.

    Args:
        suffix: The offending file extension (including the leading dot).
    """

    def __init__(self, suffix: str) -> None:
        super().__init__(
            message=f"Unsupported file type: '{suffix}'.",
            suggestions=[
                "Supported types: .jsonl, .json, .csv, .tsv, .txt.",
                "Convert your data to JSONL with 'instruction'/'output' fields.",
            ],
        )
        self.suffix = suffix


class DependencyMissingError(FlameForgeError):
    """Raised when an optional dependency required for an operation is absent.

    Args:
        package: The importable/pip name of the missing package.
        reason: Why the package is needed (e.g. "QLoRA on CUDA").
        extra_notes: Optional platform-specific guidance.
    """

    def __init__(self, package: str, reason: str, extra_notes: list[str] | None = None) -> None:
        suggestions = [f"Install it with:  pip install {package}"]
        if extra_notes:
            suggestions.extend(extra_notes)
        super().__init__(
            message=f"Missing dependency: {package}. It is required for {reason}.",
            suggestions=suggestions,
        )
        self.package = package


class ConfigurationError(FlameForgeError):
    """Raised when user-supplied or derived configuration is invalid."""


class TrainingError(FlameForgeError):
    """Raised when training fails for a reason other than out-of-memory."""


class ExportError(FlameForgeError):
    """Raised when exporting (merge/convert) a trained model fails."""
