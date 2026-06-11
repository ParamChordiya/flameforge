"""Tests for the model-loading dispatch and dependency-error handling.

Actual model downloads need a GPU/network and are out of scope; these tests
cover the parts that must work everywhere: backend dispatch, friendly
dependency errors, and the small pure helpers.
"""

from __future__ import annotations

import pytest

from flameforge.constants import Backend, FineTuningMethod
from flameforge.device.detector import DeviceInfo
from flameforge.models import cuda_loader
from flameforge.models.loader import LoadedModel, load_model
from flameforge.utils.errors import AuthenticationError, DependencyMissingError, FlameForgeError


def test_load_model_mlx_missing_dependency(mps_device: DeviceInfo) -> None:
    # mlx-lm is not installed in CI; the MLX backend must explain that clearly.
    with pytest.raises(DependencyMissingError) as exc:
        load_model("some/model", FineTuningMethod.LORA, mps_device)
    assert exc.value.package == "mlx-lm"
    assert "flameforge[mlx]" in " ".join(exc.value.suggestions)


def test_quantization_config_none_for_non_qlora() -> None:
    assert cuda_loader._quantization_config(FineTuningMethod.LORA) is None
    assert cuda_loader._quantization_config(FineTuningMethod.FULL) is None


def test_attn_implementation_safe() -> None:
    # Flash Attention is optional; the helper must degrade to None, never raise.
    assert cuda_loader._attn_implementation() in (None, "flash_attention_2")


def test_cuda_load_error_translation_auth() -> None:
    with pytest.raises(AuthenticationError):
        cuda_loader._raise_load_error("gated/model", RuntimeError("401 Client Error: gated repo"))


def test_cuda_load_error_translation_generic() -> None:
    with pytest.raises(FlameForgeError):
        cuda_loader._raise_load_error("some/model", RuntimeError("disk full"))


def test_loaded_model_dataclass() -> None:
    lm = LoadedModel(
        model=object(),
        tokenizer=object(),
        backend=Backend.MLX,
        model_id="x/y",
        quantized=True,
    )
    assert lm.backend == Backend.MLX
    assert lm.quantized is True
    assert lm.model_id == "x/y"
