"""PyTorch/transformers model loading for the CUDA backend.

This module is imported lazily by :mod:`flameforge.models.loader` so that the
heavy ``torch``/``transformers``/``bitsandbytes`` imports only happen on a CUDA
machine that is actually about to train. Every missing-dependency and auth case
is converted into an actionable FlameForge error.
"""

from __future__ import annotations

from typing import NoReturn

from flameforge.constants import Backend, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.models.loader import LoadedModel
from flameforge.utils.errors import (
    AuthenticationError,
    DependencyMissingError,
    FlameForgeError,
)
from flameforge.utils.logging import get_logger

_log = get_logger("models.cuda_loader")


def _import_transformers() -> tuple[object, object]:
    """Import and return ``(AutoModelForCausalLM, AutoTokenizer)`` or raise."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise DependencyMissingError(
            package="transformers",
            reason="loading models on CUDA",
            extra_notes=["Install the CUDA extra: pip install 'flameforge[cuda]'."],
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _quantization_config(method: FineTuningMethod) -> object | None:
    """Build a 4-bit BitsAndBytes config for QLoRA, or None for other methods.

    Raises:
        DependencyMissingError: If QLoRA is requested but bitsandbytes/torch are
            unavailable.
    """
    if method != FineTuningMethod.QLORA:
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise DependencyMissingError(
            package="bitsandbytes",
            reason="QLoRA (4-bit) training on CUDA",
            extra_notes=[
                "Install it with: pip install bitsandbytes",
                "bitsandbytes requires Linux/Windows with an NVIDIA GPU.",
                "On Apple Silicon, FlameForge uses MLX instead (auto-detected).",
            ],
        ) from exc
    config: object = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    return config


def _attn_implementation() -> str | None:
    """Return "flash_attention_2" if Flash Attention is installed, else None."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return None
    return "flash_attention_2"


def load(
    model_id: str,
    method: FineTuningMethod,
    device: DeviceInfo,
    token: str | None = None,
) -> LoadedModel:
    """Load a model + tokenizer on CUDA, applying quantization for QLoRA.

    Args:
        model_id: HuggingFace id or local path.
        method: The fine-tuning method.
        device: The detected CUDA device.
        token: Optional HuggingFace token.

    Returns:
        A :class:`LoadedModel` wrapping the torch model and tokenizer.

    Raises:
        DependencyMissingError: If required libraries are missing.
        AuthenticationError: If the model is gated and access is denied.
        FlameForgeError: For other load failures.
    """
    auto_model, auto_tokenizer = _import_transformers()
    import torch

    quant_config = _quantization_config(method)
    dtype = torch.bfloat16 if device.compute_capability and device.compute_capability >= "8.0" else torch.float16

    kwargs: dict[str, object] = {"token": token, "torch_dtype": dtype, "device_map": "auto"}
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    attn = _attn_implementation()
    if attn is not None:
        kwargs["attn_implementation"] = attn
        _log.info("Flash Attention 2 enabled")

    _log.info("Loading CUDA model %s (method=%s, dtype=%s)", model_id, method.value, dtype)
    try:
        model = auto_model.from_pretrained(model_id, **kwargs)  # type: ignore[attr-defined]
        tokenizer = auto_tokenizer.from_pretrained(model_id, token=token)  # type: ignore[attr-defined]
    except Exception as exc:  # transformers raises many error types; normalize them.
        _raise_load_error(model_id, exc)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore[attr-defined]

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        backend=Backend.PYTORCH,
        model_id=model_id,
        quantized=quant_config is not None,
    )


def _raise_load_error(model_id: str, exc: Exception) -> NoReturn:
    """Translate a transformers load exception into a FlameForge error."""
    text = str(exc).lower()
    if "401" in text or "gated" in text or "authentication" in text or "access to model" in text:
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
            "Verify the model id is correct and downloadable.",
            "Check your internet connection and available disk space.",
        ],
        details=str(exc),
    ) from exc
