"""Render normalized examples into model-ready training strings and token counts.

The :class:`Formatter` prefers a model tokenizer's own ``apply_chat_template``
(the most compatible option) and falls back to FlameForge's hardcoded
:mod:`flameforge.data.templates` when a tokenizer is unavailable or lacks a
template. Token counts use the tokenizer when present and a fast character-based
heuristic otherwise, so the data-preview and validation screens work even before
a model is downloaded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flameforge.data.loader import Example
from flameforge.data.templates import get_template, resolve_template_key
from flameforge.utils.logging import get_logger

_log = get_logger("data.formatter")

# Rough chars-per-token used when no tokenizer is available. English text is
# ~4 characters per token on average for modern BPE tokenizers.
_CHARS_PER_TOKEN = 4


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal structural interface for a HuggingFace-style tokenizer."""

    def encode(self, text: str) -> list[int]:
        """Return token ids for ``text``."""
        ...


class Formatter:
    """Formats normalized examples into training strings and counts tokens.

    Args:
        template_key: Explicit chat-template key; resolved from ``family`` if None.
        family: Model family used to resolve a template when no key is given.
        tokenizer: Optional tokenizer. If it exposes ``apply_chat_template`` that
            is used for chat examples; otherwise its ``encode`` is used for token
            counting and the hardcoded template is used for rendering.
    """

    def __init__(
        self,
        template_key: str | None = None,
        family: str | None = None,
        tokenizer: object | None = None,
    ) -> None:
        self.template_key = resolve_template_key(template_key=template_key, family=family)
        self.tokenizer = tokenizer
        self._template = get_template(self.template_key)
        self._has_chat_template = hasattr(tokenizer, "apply_chat_template")

    def format_example(self, example: Example) -> str:
        """Render one example to the exact string fed to the model.

        Args:
            example: A normalized example (chat or text).

        Returns:
            The rendered training string.
        """
        if not example.is_chat:
            return example.text or ""

        assert example.messages is not None  # narrowed by is_chat
        if self._has_chat_template and self.tokenizer is not None:
            try:
                payload = [{"role": m.role, "content": m.content} for m in example.messages]
                rendered = self.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                    payload, tokenize=False, add_generation_prompt=False
                )
                return str(rendered)
            except Exception as exc:  # pragma: no cover - tokenizer-specific
                # Never let a tokenizer quirk crash formatting; fall back cleanly.
                _log.warning("tokenizer.apply_chat_template failed (%s); using fallback template", exc)

        template = self._template
        if template is None:  # pragma: no cover - resolve guarantees a template
            return "\n".join(f"{m.role}: {m.content}" for m in example.messages)
        return template.render([(m.role, m.content) for m in example.messages])

    def count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` using the tokenizer, or a heuristic.

        Args:
            text: The string to measure.

        Returns:
            The token count (exact with a tokenizer, estimated without one).
        """
        if isinstance(self.tokenizer, TokenizerLike):
            try:
                return len(self.tokenizer.encode(text))
            except Exception:  # pragma: no cover - tokenizer-specific
                pass
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def format_with_token_count(self, example: Example) -> tuple[str, int]:
        """Render an example and count its tokens in one call.

        Args:
            example: The example to render.

        Returns:
            A tuple of (rendered string, token count).
        """
        text = self.format_example(example)
        return text, self.count_tokens(text)

    def to_mlx_record(self, example: Example) -> dict[str, str]:
        """Render an example into mlx-lm's expected ``{"text": ...}`` record.

        Args:
            example: The example to render.

        Returns:
            A dict with a single ``text`` key, ready to serialize as JSONL.
        """
        return {"text": self.format_example(example)}
