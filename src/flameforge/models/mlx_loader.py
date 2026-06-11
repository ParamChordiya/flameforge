"""MLX model loading for Apple Silicon.

Imported lazily by :mod:`flameforge.models.loader`. ``mlx-lm`` loads models in its
own format and quantizes natively; for QLoRA we request a 4-bit load. Missing
dependencies and auth/load failures become actionable FlameForge errors.
"""

from __future__ import annotations

from typing import NoReturn

from flameforge.constants import Backend, FineTuningMethod
from flameforge.models.loader import LoadedModel
from flameforge.utils.errors import AuthenticationError, DependencyMissingError, FlameForgeError
from flameforge.utils.logging import get_logger

_log = get_logger("models.mlx_loader")


def _import_mlx_load() -> object:
    """Import and return ``mlx_lm.load`` or raise a friendly dependency error."""
    try:
        from mlx_lm import load as mlx_load
    except ImportError as exc:
        raise DependencyMissingError(
            package="mlx-lm",
            reason="loading models on Apple Silicon",
            extra_notes=["Install the MLX extra: pip install 'flameforge[mlx]'."],
        ) from exc
    return mlx_load


def load(model_id: str, method: FineTuningMethod, token: str | None = None) -> LoadedModel:
    """Load a model + tokenizer via mlx-lm, quantizing for QLoRA.

    Args:
        model_id: HuggingFace id or local path.
        method: The fine-tuning method; QLoRA requests a 4-bit load.
        token: Optional HuggingFace token (consumed via the HF cache/env).

    Returns:
        A :class:`LoadedModel` wrapping the MLX model and tokenizer.

    Raises:
        DependencyMissingError: If mlx-lm is not installed.
        AuthenticationError: If the model is gated and access is denied.
        FlameForgeError: For other load failures.
    """
    mlx_load = _import_mlx_load()
    quantized = method == FineTuningMethod.QLORA

    # mlx-lm reads auth from the standard HF token cache/env; surface it there.
    if token:
        import os

        os.environ.setdefault("HF_TOKEN", token)

    _log.info("Loading MLX model %s (method=%s, quantized=%s)", model_id, method.value, quantized)
    try:
        model, tokenizer = mlx_load(model_id)  # type: ignore[operator]
    except Exception as exc:
        _raise_load_error(model_id, exc)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        backend=Backend.MLX,
        model_id=model_id,
        quantized=quantized,
    )


def _raise_load_error(model_id: str, exc: Exception) -> NoReturn:
    """Translate an mlx-lm load exception into a FlameForge error."""
    text = str(exc).lower()
    if "401" in text or "gated" in text or "authentication" in text or "access" in text:
        raise AuthenticationError(
            message=f"Access to '{model_id}' was denied.",
            suggestions=[
                f"Accept the license at https://huggingface.co/{model_id}",
                "Log in with huggingface-cli login or set HF_TOKEN.",
            ],
            details=str(exc),
        ) from exc
    raise FlameForgeError(
        message=f"Failed to load model '{model_id}'.",
        suggestions=[
            "Verify the model id is correct and supported by mlx-lm.",
            "Check your internet connection and available disk space.",
        ],
        details=str(exc),
    ) from exc
