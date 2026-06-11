"""Tests for HuggingFace helpers: token discovery, saving, and error mapping."""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

import flameforge.utils.hf_utils as hf
from flameforge.utils.errors import AuthenticationError, ModelNotFoundError


def test_find_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "  hf_envtoken  ")
    assert hf.find_hf_token() == "hf_envtoken"


def test_find_token_prefers_env_over_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "token"
    cache.write_text("hf_cachetoken", encoding="utf-8")
    monkeypatch.setattr(hf, "_TOKEN_CACHE_PATH", cache)
    monkeypatch.setenv("HF_TOKEN", "hf_envtoken")
    assert hf.find_hf_token() == "hf_envtoken"


def test_find_token_from_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in hf._TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    cache = tmp_path / "token"
    cache.write_text("hf_cachetoken\n", encoding="utf-8")
    monkeypatch.setattr(hf, "_TOKEN_CACHE_PATH", cache)
    assert hf.find_hf_token() == "hf_cachetoken"


def test_find_token_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in hf._TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(hf, "_TOKEN_CACHE_PATH", tmp_path / "missing")
    assert hf.find_hf_token() is None


def test_save_token_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "nested" / "token"
    monkeypatch.setattr(hf, "_TOKEN_CACHE_PATH", cache)
    path = hf.save_hf_token("  hf_secret  ")
    assert path == cache
    assert cache.read_text(encoding="utf-8") == "hf_secret"


def test_is_local_model_path(tmp_path: Path) -> None:
    assert hf.is_local_model_path(str(tmp_path)) is True
    assert hf.is_local_model_path(str(tmp_path / "nope")) is False


def test_search_empty_query_returns_empty() -> None:
    assert hf.search_hub_models("   ") == []


class _FakeApi:
    """Stand-in for HfApi that raises a preconfigured error from model_info."""

    def __init__(self, error: Exception | None = None, **_: object) -> None:
        self._error = error

    def model_info(self, *_: object, **__: object) -> object:
        if self._error is not None:
            raise self._error
        return object()


def _make_error(cls: type[Exception]) -> Exception:
    """Instantiate a HuggingFace Hub error without invoking its (version-varying)
    ``__init__``, whose required arguments differ across huggingface_hub releases.
    """
    return cls.__new__(cls)


def test_ensure_access_gated_maps_to_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _make_error(GatedRepoError)
    monkeypatch.setattr(hf, "HfApi", lambda **kw: _FakeApi(err))
    with pytest.raises(AuthenticationError) as exc:
        hf.ensure_model_access("meta-llama/Llama-3.1-8B-Instruct", token="hf_x")
    assert "license" in str(exc.value).lower()


def test_ensure_access_missing_with_token_maps_to_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _make_error(RepositoryNotFoundError)
    monkeypatch.setattr(hf, "HfApi", lambda **kw: _FakeApi(err))
    with pytest.raises(ModelNotFoundError):
        hf.ensure_model_access("nope/model", token="hf_x")


def test_ensure_access_missing_without_token_suggests_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _make_error(RepositoryNotFoundError)
    monkeypatch.setattr(hf, "HfApi", lambda **kw: _FakeApi(err))
    monkeypatch.setattr(hf, "find_hf_token", lambda: None)
    with pytest.raises(AuthenticationError):
        hf.ensure_model_access("private/model", token=None)


def test_ensure_access_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "HfApi", lambda **kw: _FakeApi(None))
    hf.ensure_model_access("public/model", token=None)  # should not raise
