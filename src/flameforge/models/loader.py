"""Unified model-loading interface dispatching to the CUDA or MLX backend.

Callers use :func:`load_model` and :func:`load_tokenizer` without caring which
backend is active. The heavy ML libraries (torch/transformers/mlx) are imported
lazily inside the backend modules, so importing this module is cheap and never
fails just because a backend's extras are not installed — a missing backend turns
into a friendly :class:`~flameforge.utils.errors.DependencyMissingError` only when
a load is actually attempted.
"""

from __future__ import annotations

from dataclasses import dataclass

from flameforge.constants import Backend, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.utils.errors import DependencyMissingError
from flameforge.utils.logging import get_logger

_log = get_logger("models.loader")


@dataclass
class LoadedModel:
    """A loaded model and tokenizer with the backend that produced them.

    Attributes:
        model: The backend-specific model object.
        tokenizer: The backend-specific tokenizer object.
        backend: Which backend loaded the model.
        model_id: The id/path the model was loaded from.
        quantized: Whether the base weights were loaded quantized (QLoRA).
    """

    model: object
    tokenizer: object
    backend: Backend
    model_id: str
    quantized: bool


def load_model(
    model_id: str,
    method: FineTuningMethod,
    device: DeviceInfo,
    token: str | None = None,
) -> LoadedModel:
    """Load a base model and tokenizer for the active backend.

    Args:
        model_id: HuggingFace id or local path.
        method: The fine-tuning method (controls quantization for QLoRA).
        device: The detected device, whose ``backend`` selects the loader.
        token: Optional HuggingFace token for gated models.

    Returns:
        A :class:`LoadedModel`.

    Raises:
        DependencyMissingError: If the backend's libraries are not installed.
        FlameForgeError: For auth/memory/loading failures (raised by backends).
    """
    if device.backend == Backend.MLX:
        from flameforge.models import mlx_loader

        return mlx_loader.load(model_id, method, token=token)
    from flameforge.models import cuda_loader

    return cuda_loader.load(model_id, method, device, token=token)


def load_tokenizer(model_id: str, token: str | None = None) -> object:
    """Load just the tokenizer for a model, for formatting and token counting.

    The tokenizer is light to load and lets FlameForge render data previews and
    compute exact token counts before committing to a full model download.

    Args:
        model_id: HuggingFace id or local path.
        token: Optional HuggingFace token for gated models.

    Returns:
        A tokenizer object exposing ``encode`` and (usually) ``apply_chat_template``.

    Raises:
        DependencyMissingError: If ``transformers`` is not installed.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise DependencyMissingError(
            package="transformers",
            reason="loading tokenizers",
            extra_notes=["Install a backend, e.g. pip install 'flameforge[mlx]' or 'flameforge[cuda]'."],
        ) from exc
    _log.info("Loading tokenizer for %s", model_id)
    return AutoTokenizer.from_pretrained(model_id, token=token)
