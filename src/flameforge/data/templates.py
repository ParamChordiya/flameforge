"""Chat templates per model family, with a single-source rendering helper.

FlameForge always prefers a model tokenizer's own ``apply_chat_template`` (see
:mod:`flameforge.data.formatter`). These hardcoded templates are the fallback for
tokenizers that ship without one, and they let the data-preview screen show a
faithful rendering before any model is downloaded.

Each template is expressed as a list of *segments* keyed by role so that
multi-turn conversations render correctly — a single ``{user}``/``{assistant}``
format string cannot represent more than one turn.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTemplate:
    """A role-segmented chat template for one model family.

    Attributes:
        key: The template identifier (e.g. "llama3").
        bos: Optional text emitted once at the very start of the sequence.
        system: Format string for a system turn, with a ``{content}`` slot, or
            None if the family does not support system turns.
        user: Format string for a user turn (``{content}`` slot).
        assistant: Format string for an assistant turn (``{content}`` slot).
    """

    key: str
    bos: str
    system: str | None
    user: str
    assistant: str

    def render(self, messages: list[tuple[str, str]]) -> str:
        """Render a conversation to a single training string.

        Args:
            messages: Ordered ``(role, content)`` pairs. Recognised roles are
                ``system``, ``user``, and ``assistant``. A system message is
                merged into the first user turn for families without system
                support.

        Returns:
            The fully rendered conversation string.
        """
        out = [self.bos]
        pending_system: str | None = None
        for role, content in messages:
            if role == "system":
                if self.system is not None:
                    out.append(self.system.format(content=content))
                else:
                    pending_system = content
            elif role == "user":
                text = content
                if pending_system:
                    text = f"{pending_system}\n\n{content}"
                    pending_system = None
                out.append(self.user.format(content=text))
            elif role == "assistant":
                out.append(self.assistant.format(content=content))
        return "".join(out)


CHAT_TEMPLATES: dict[str, ChatTemplate] = {
    "llama3": ChatTemplate(
        key="llama3",
        bos="<|begin_of_text|>",
        system="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        user="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        assistant="<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    ),
    "mistral": ChatTemplate(
        key="mistral",
        bos="<s>",
        system=None,  # Mistral has no system role; merged into the first user turn.
        user="[INST] {content} [/INST]",
        assistant=" {content}</s>",
    ),
    "chatml": ChatTemplate(
        key="chatml",
        bos="",
        system="<|im_start|>system\n{content}<|im_end|>\n",
        user="<|im_start|>user\n{content}<|im_end|>\n",
        assistant="<|im_start|>assistant\n{content}<|im_end|>\n",
    ),
    "gemma": ChatTemplate(
        key="gemma",
        bos="<bos>",
        system=None,  # Gemma folds system content into the first user turn.
        user="<start_of_turn>user\n{content}<end_of_turn>\n",
        assistant="<start_of_turn>model\n{content}<end_of_turn>\n",
    ),
    "phi3": ChatTemplate(
        key="phi3",
        bos="",
        system="<|system|>\n{content}<|end|>\n",
        user="<|user|>\n{content}<|end|>\n",
        assistant="<|assistant|>\n{content}<|end|>\n",
    ),
    "qwen": ChatTemplate(
        key="qwen",
        bos="",
        system="<|im_start|>system\n{content}<|im_end|>\n",
        user="<|im_start|>user\n{content}<|im_end|>\n",
        assistant="<|im_start|>assistant\n{content}<|im_end|>\n",
    ),
}

# Map model-family keys (from the registry) to a template key.
FAMILY_TO_TEMPLATE: dict[str, str] = {
    "llama": "llama3",
    "mistral": "mistral",
    "gemma": "gemma",
    "qwen": "qwen",
    "phi": "phi3",
}

DEFAULT_TEMPLATE_KEY = "chatml"


def get_template(key: str) -> ChatTemplate | None:
    """Return the :class:`ChatTemplate` for a template key, or None if unknown.

    Args:
        key: A template key such as "llama3" or "qwen".

    Returns:
        The matching template, or None.
    """
    return CHAT_TEMPLATES.get(key)


def resolve_template_key(*, template_key: str | None = None, family: str | None = None) -> str:
    """Resolve the best template key from an explicit key and/or model family.

    Args:
        template_key: An explicit template key, if the caller already knows it.
        family: The model family (e.g. "llama"), used when no key is given.

    Returns:
        A valid template key, falling back to :data:`DEFAULT_TEMPLATE_KEY`.
    """
    if template_key and template_key in CHAT_TEMPLATES:
        return template_key
    if family and family in FAMILY_TO_TEMPLATE:
        return FAMILY_TO_TEMPLATE[family]
    return DEFAULT_TEMPLATE_KEY
