"""Tests for the export modules (merge + GGUF conversion)."""

from __future__ import annotations

from pathlib import Path

import pytest

from flameforge.constants import Backend
from flameforge.export.convert import convert_to_gguf, find_converter
from flameforge.export.merge import _atomic_dir, merge_adapter
from flameforge.utils.errors import DependencyMissingError, ExportError


def test_merge_missing_adapter_raises(tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        merge_adapter("x/y", tmp_path / "missing", tmp_path / "out", Backend.MLX)


def test_atomic_dir_commits_on_success(tmp_path: Path) -> None:
    target = tmp_path / "result"
    with _atomic_dir(target) as staging:
        (staging / "file.txt").write_text("hi", encoding="utf-8")
    assert target.is_dir()
    assert (target / "file.txt").read_text(encoding="utf-8") == "hi"


def test_atomic_dir_rolls_back_on_error(tmp_path: Path) -> None:
    target = tmp_path / "result"
    with pytest.raises(RuntimeError), _atomic_dir(target) as staging:
        (staging / "file.txt").write_text("hi", encoding="utf-8")
        raise RuntimeError("boom")
    assert not target.exists()
    # No leftover temp directories.
    assert list(tmp_path.iterdir()) == []


def test_atomic_dir_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "result"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    with _atomic_dir(target) as staging:
        (staging / "new.txt").write_text("new", encoding="utf-8")
    assert (target / "new.txt").exists()
    assert not (target / "old.txt").exists()


def test_gguf_unknown_quant_raises(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    with pytest.raises(ExportError):
        convert_to_gguf(tmp_path / "model", tmp_path / "out.gguf", quantization="Q3_BOGUS")


def test_gguf_missing_model_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        convert_to_gguf(tmp_path / "nope", tmp_path / "out.gguf", quantization="Q4_K_M")


def test_gguf_without_converter_raises_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.delenv("FLAMEFORGE_LLAMACPP", raising=False)
    monkeypatch.setattr("flameforge.export.convert.find_converter", lambda: None)
    with pytest.raises(DependencyMissingError):
        convert_to_gguf(model_dir, tmp_path / "out.gguf", quantization="Q4_K_M")


def test_find_converter_via_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# converter", encoding="utf-8")
    monkeypatch.setenv("FLAMEFORGE_LLAMACPP", str(tmp_path))
    assert find_converter() == script
