"""Merge trained LoRA/DoRA adapters into a standalone base model.

Both backends are supported: PyTorch/PEFT merges in-process, while MLX delegates
to ``mlx_lm.fuse`` (its documented, well-tested fusing path). Missing
dependencies become actionable :class:`DependencyMissingError`s, and the merged
model is written atomically — to a temp directory that is renamed into place only
once the whole save succeeds — so a failure never leaves a half-written model.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from flameforge.constants import Backend
from flameforge.utils.errors import DependencyMissingError, ExportError
from flameforge.utils.logging import get_logger

_log = get_logger("export.merge")


def merge_adapter(
    base_model_id: str,
    adapter_path: str | Path,
    output_path: str | Path,
    backend: Backend,
    token: str | None = None,
) -> Path:
    """Merge an adapter into its base model and save the result.

    Args:
        base_model_id: The base model's HuggingFace id or local path.
        adapter_path: Directory containing the trained adapter weights.
        output_path: Destination directory for the merged model.
        backend: Which backend produced the adapter.
        token: Optional HuggingFace token for a gated base model.

    Returns:
        The path to the merged model directory.

    Raises:
        DependencyMissingError: If the backend's libraries are missing.
        ExportError: If merging or saving fails.
    """
    out = Path(output_path)
    adapter = Path(adapter_path)
    if not adapter.exists():
        raise ExportError(
            message=f"Adapter not found at '{adapter}'.",
            suggestions=["Train a model first, or point at the correct output directory."],
        )

    if backend == Backend.MLX:
        return _merge_mlx(base_model_id, adapter, out)
    return _merge_cuda(base_model_id, adapter, out, token)


def _merge_cuda(base_model_id: str, adapter: Path, output_path: Path, token: str | None) -> Path:
    """Merge with PyTorch + PEFT, saving atomically."""
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise DependencyMissingError(
            package="peft transformers",
            reason="merging adapters on CUDA",
            extra_notes=["Install the CUDA extra: pip install 'flameforge[cuda]'."],
        ) from exc

    _log.info("Merging adapter %s into %s", adapter, base_model_id)
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model_id, token=token)
        model = PeftModel.from_pretrained(base, str(adapter))
        merged = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)
        with _atomic_dir(output_path) as staging:
            merged.save_pretrained(staging)
            tokenizer.save_pretrained(staging)
    except Exception as exc:
        raise ExportError(
            message="Failed to merge the adapter into the base model.",
            suggestions=["Ensure the base model matches the one used for training."],
            details=str(exc),
        ) from exc
    return output_path


def _merge_mlx(base_model_id: str, adapter: Path, output_path: Path) -> Path:
    """Merge with mlx-lm's ``fuse`` CLI, saving atomically."""
    if not _module_available("mlx_lm"):
        raise DependencyMissingError(
            package="mlx-lm",
            reason="fusing adapters on Apple Silicon",
            extra_notes=["Install the MLX extra: pip install 'flameforge[mlx]'."],
        )
    _log.info("Fusing MLX adapter %s into %s", adapter, base_model_id)
    with _atomic_dir(output_path) as staging:
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            base_model_id,
            "--adapter-path",
            str(adapter),
            "--save-path",
            str(staging),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ExportError(
                message="mlx-lm failed to fuse the adapter.",
                suggestions=["Check that the base model and adapter are compatible."],
                details=result.stderr.strip() or result.stdout.strip(),
            )
    return output_path


def _module_available(name: str) -> bool:
    """Return whether a module can be imported without importing it."""
    import importlib.util

    return importlib.util.find_spec(name) is not None


class _atomic_dir:
    """Context manager yielding a temp dir that is renamed to ``target`` on success.

    If the body raises, the temp directory is removed and ``target`` is left
    untouched, so an export is all-or-nothing.
    """

    def __init__(self, target: Path) -> None:
        self._target = target
        self._tmp: Path | None = None

    def __enter__(self) -> Path:
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = Path(tempfile.mkdtemp(prefix=f".{self._target.name}-", dir=self._target.parent))
        return self._tmp

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._tmp is not None
        if exc_type is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            return
        if self._target.exists():
            shutil.rmtree(self._target)
        self._tmp.rename(self._target)
