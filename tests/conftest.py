"""Shared pytest fixtures for the FlameForge test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameforge.constants import Backend, DeviceType
from flameforge.device.detector import DeviceInfo


@pytest.fixture
def cuda_device() -> DeviceInfo:
    """A synthetic CUDA device with a 24 GB GPU (21.6 GB budget)."""
    return DeviceInfo(
        device_type=DeviceType.CUDA,
        device_name="NVIDIA RTX 4090",
        total_memory_gb=24.0,
        memory_budget_gb=21.6,
        compute_capability="8.9",
        gpu_count=1,
        backend=Backend.PYTORCH,
    )


@pytest.fixture
def mps_device() -> DeviceInfo:
    """A synthetic Apple Silicon device with 16 GB unified memory (11.2 GB budget)."""
    return DeviceInfo(
        device_type=DeviceType.MPS,
        device_name="Apple M2 Pro",
        total_memory_gb=16.0,
        memory_budget_gb=11.2,
        compute_capability=None,
        gpu_count=1,
        backend=Backend.MLX,
    )


@pytest.fixture
def cpu_device() -> DeviceInfo:
    """A synthetic CPU-only device with 8 GB RAM (4 GB budget)."""
    return DeviceInfo(
        device_type=DeviceType.CPU,
        device_name="Generic CPU",
        total_memory_gb=8.0,
        memory_budget_gb=4.0,
        compute_capability=None,
        gpu_count=0,
        backend=Backend.PYTORCH,
    )


@pytest.fixture
def alpaca_file(tmp_path: Path) -> Path:
    """Write a small Alpaca-format JSONL file and return its path."""
    rows = [
        {"instruction": "Say hello.", "input": "", "output": "Hello!"},
        {"instruction": "Add two numbers.", "input": "2 and 3", "output": "5"},
        {"instruction": "Capital of France?", "input": "", "output": "Paris"},
    ]
    path = tmp_path / "alpaca.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def chat_file(tmp_path: Path) -> Path:
    """Write a small conversational JSONL file and return its path."""
    rows = [
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},
        {
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
    ]
    path = tmp_path / "chat.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    """Write a small raw-text file (two documents) and return its path."""
    path = tmp_path / "corpus.txt"
    path.write_text("First document paragraph.\n\nSecond document paragraph.", encoding="utf-8")
    return path


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """Write a small CSV file with prompt/response-like columns."""
    path = tmp_path / "data.csv"
    path.write_text(
        "question,answer\nWhat is 2+2?,4\nCapital of Japan?,Tokyo\n",
        encoding="utf-8",
    )
    return path
