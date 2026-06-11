"""Auto-detect the compute device and pick the appropriate training backend.

Detection is deliberately dependency-light: ``torch`` is imported lazily and its
absence is treated as "no CUDA" rather than an error, so FlameForge can run its
TUI and data pipeline even before the heavy ML extras are installed. On Apple
Silicon we shell out to ``sysctl`` for accurate unified-memory and chip-name
information.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass

import psutil

from flameforge.constants import Backend, DeviceType
from flameforge.device.memory import calculate_memory_budget


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable description of the detected compute device.

    Attributes:
        device_type: One of CUDA, MPS, or CPU.
        device_name: Human-readable device name (e.g. "Apple M2 Pro").
        total_memory_gb: Total GPU/unified/system memory in gigabytes.
        memory_budget_gb: Safe training budget in gigabytes.
        compute_capability: CUDA compute capability (e.g. "8.9") or None.
        gpu_count: Number of GPUs detected (1 for MPS/CPU).
        backend: The training backend to use (PyTorch or MLX).
    """

    device_type: DeviceType
    device_name: str
    total_memory_gb: float
    memory_budget_gb: float
    compute_capability: str | None
    gpu_count: int
    backend: Backend

    @property
    def is_cpu_only(self) -> bool:
        """Whether training will fall back to (very slow) CPU execution."""
        return self.device_type == DeviceType.CPU

    def summary_lines(self) -> list[str]:
        """Return human-readable lines describing the device for the welcome screen."""
        lines = [
            f"Device:   {self.device_name}",
            f"Type:     {self.device_type.value.upper()}",
            f"Backend:  {self.backend.value}",
            f"Memory:   {self.total_memory_gb:.1f} GB total",
            f"Budget:   {self.memory_budget_gb:.1f} GB safe training budget",
        ]
        if self.gpu_count > 1:
            lines.append(f"GPUs:     {self.gpu_count} (single-GPU training by default)")
        if self.compute_capability:
            lines.append(f"Compute:  {self.compute_capability}")
        return lines


def _detect_cuda() -> DeviceInfo | None:
    """Detect an NVIDIA CUDA device via torch, returning None if unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    gpu_count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    capability = f"{props.major}.{props.minor}"
    info = DeviceInfo(
        device_type=DeviceType.CUDA,
        device_name=props.name,
        total_memory_gb=total_gb,
        memory_budget_gb=0.0,  # filled below
        compute_capability=capability,
        gpu_count=gpu_count,
        backend=Backend.PYTORCH,
    )
    budget = calculate_memory_budget(info)
    return _with_budget(info, budget)


def _sysctl(key: str) -> str | None:
    """Return the trimmed string value of a macOS sysctl key, or None on failure."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    value = out.stdout.strip()
    return value or None


def _detect_apple_silicon() -> DeviceInfo | None:
    """Detect an Apple Silicon (MPS) device, returning None if not applicable."""
    if sys.platform != "darwin":
        return None
    if "arm" not in platform.processor().lower() and platform.machine().lower() != "arm64":
        return None

    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "Apple Silicon"
    memsize = _sysctl("hw.memsize")
    total_gb = int(memsize) / 1e9 if memsize is not None else psutil.virtual_memory().total / 1e9

    info = DeviceInfo(
        device_type=DeviceType.MPS,
        device_name=chip,
        total_memory_gb=total_gb,
        memory_budget_gb=0.0,
        compute_capability=None,
        gpu_count=1,
        backend=Backend.MLX,
    )
    budget = calculate_memory_budget(info)
    return _with_budget(info, budget)


def _detect_cpu() -> DeviceInfo:
    """Fallback CPU device using system RAM (training here is extremely slow)."""
    total_gb = psutil.virtual_memory().total / 1e9
    info = DeviceInfo(
        device_type=DeviceType.CPU,
        device_name=platform.processor() or platform.machine() or "CPU",
        total_memory_gb=total_gb,
        memory_budget_gb=0.0,
        compute_capability=None,
        gpu_count=0,
        backend=Backend.PYTORCH,
    )
    budget = calculate_memory_budget(info)
    return _with_budget(info, budget)


def _with_budget(info: DeviceInfo, budget: float) -> DeviceInfo:
    """Return a copy of ``info`` with ``memory_budget_gb`` set to ``budget``."""
    return DeviceInfo(
        device_type=info.device_type,
        device_name=info.device_name,
        total_memory_gb=info.total_memory_gb,
        memory_budget_gb=budget,
        compute_capability=info.compute_capability,
        gpu_count=info.gpu_count,
        backend=info.backend,
    )


def detect_device(user_cap_gb: float | None = None) -> DeviceInfo:
    """Detect the best available compute device.

    Detection priority is CUDA, then Apple Silicon (MPS), then CPU. A user memory
    cap, if provided, is applied to whichever device is selected.

    Args:
        user_cap_gb: Optional hard cap on the training memory budget, in GB.

    Returns:
        A populated :class:`DeviceInfo` describing the selected device.
    """
    info = _detect_cuda() or _detect_apple_silicon() or _detect_cpu()
    if user_cap_gb is not None:
        info = _with_budget(info, calculate_memory_budget(info, user_cap_gb))
    return info
