"""Convert a merged model to GGUF for use with llama.cpp / Ollama.

GGUF conversion relies on llama.cpp's ``convert_hf_to_gguf.py`` script, which is
not a pip-installable library. We locate it via (in order) the
``FLAMEFORGE_LLAMACPP`` environment variable, a script on ``PATH``, or an
installed ``gguf`` package's tooling, and otherwise raise a clear
:class:`DependencyMissingError` telling the user exactly how to get it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from flameforge.constants import GGUF_QUANT_LEVELS
from flameforge.utils.errors import DependencyMissingError, ExportError
from flameforge.utils.logging import get_logger

_log = get_logger("export.convert")

_CONVERT_SCRIPT_NAMES = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")


def find_converter() -> Path | None:
    """Locate llama.cpp's HF→GGUF conversion script.

    Returns:
        The path to a conversion script, or None if none can be found.
    """
    env = os.environ.get("FLAMEFORGE_LLAMACPP")
    if env:
        root = Path(env)
        for name in _CONVERT_SCRIPT_NAMES:
            candidate = root / name
            if candidate.is_file():
                return candidate
    for name in _CONVERT_SCRIPT_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def convert_to_gguf(
    model_path: str | Path,
    output_path: str | Path,
    quantization: str = "Q4_K_M",
) -> Path:
    """Convert a merged HuggingFace model directory to a GGUF file.

    Args:
        model_path: Directory containing the merged model.
        output_path: Destination ``.gguf`` file path.
        quantization: One of :data:`flameforge.constants.GGUF_QUANT_LEVELS`.

    Returns:
        The path to the written GGUF file.

    Raises:
        ExportError: If inputs are invalid or conversion fails.
        DependencyMissingError: If no llama.cpp converter can be found.
    """
    if quantization not in GGUF_QUANT_LEVELS:
        raise ExportError(
            message=f"Unknown GGUF quantization '{quantization}'.",
            suggestions=[f"Choose one of: {', '.join(GGUF_QUANT_LEVELS)}."],
        )
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise ExportError(
            message=f"Merged model directory not found: '{model_dir}'.",
            suggestions=["Export a merged model first, then convert it to GGUF."],
        )

    converter = find_converter()
    if converter is None:
        raise DependencyMissingError(
            package="llama.cpp",
            reason="GGUF conversion",
            extra_notes=[
                "Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp",
                "Then set FLAMEFORGE_LLAMACPP=/path/to/llama.cpp so FlameForge can find it.",
                "Its convert_hf_to_gguf.py does the conversion.",
            ],
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outfile",
        str(out),
        "--outtype",
        _outtype_for(quantization),
    ]
    _log.info("Converting to GGUF: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ExportError(
            message="GGUF conversion failed.",
            suggestions=["Check that llama.cpp is up to date and supports this architecture."],
            details=result.stderr.strip() or result.stdout.strip(),
        )
    return out


def _outtype_for(quantization: str) -> str:
    """Map a GGUF quant level to a converter ``--outtype`` value.

    The converter writes f16 then a separate quantize step refines it; for the
    common levels we pass the closest base type it understands.
    """
    if quantization == "F16":
        return "f16"
    if quantization == "Q8_0":
        return "q8_0"
    # K-quants are produced from an f16 base by llama.cpp's quantize tool; the
    # convert script emits f16 which the user can quantize further.
    return "f16"
