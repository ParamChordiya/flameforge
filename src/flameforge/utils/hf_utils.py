"""HuggingFace Hub helpers: token discovery, validation, search, and access.

The guiding principle is *no cryptic 401s*. Auth is discovered from the
environment or the standard token cache; when a model is gated we diagnose
exactly what the user must do (accept a license, create a token) and surface it
through :class:`~flameforge.utils.errors.AuthenticationError`. Network and Hub
errors are likewise caught and turned into actionable messages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from flameforge.utils.errors import AuthenticationError, ModelNotFoundError
from flameforge.utils.logging import get_logger

_log = get_logger("hf")

# Standard locations the HuggingFace stack reads tokens from.
_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
_TOKEN_CACHE_PATH = Path.home() / ".cache" / "huggingface" / "token"


@dataclass(frozen=True)
class HubModelResult:
    """A lightweight search result from the Hub.

    Attributes:
        model_id: The repository id.
        downloads: Recent download count (0 if unknown).
        likes: Like count (0 if unknown).
        gated: Whether the repo is gated behind a license.
    """

    model_id: str
    downloads: int
    likes: int
    gated: bool


def find_hf_token() -> str | None:
    """Discover a HuggingFace token from the environment or token cache.

    Returns:
        The token string if found (env vars take precedence over the cache file),
        otherwise None.
    """
    for var in _TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value.strip()
    if _TOKEN_CACHE_PATH.is_file():
        token = _TOKEN_CACHE_PATH.read_text(encoding="utf-8").strip()
        return token or None
    return None


def save_hf_token(token: str) -> Path:
    """Persist a token to the standard HuggingFace cache location.

    Args:
        token: The token to store.

    Returns:
        The path the token was written to.

    Raises:
        AuthenticationError: If the token cache cannot be written.
    """
    try:
        _TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_CACHE_PATH.write_text(token.strip(), encoding="utf-8")
        _TOKEN_CACHE_PATH.chmod(0o600)
    except OSError as exc:
        raise AuthenticationError(
            message="Could not save your HuggingFace token.",
            suggestions=[
                f"Check permissions on {_TOKEN_CACHE_PATH.parent}.",
                "Or set the HF_TOKEN environment variable instead.",
            ],
            details=str(exc),
        ) from exc
    return _TOKEN_CACHE_PATH


def validate_token(token: str) -> str:
    """Validate a token against the Hub and return the associated username.

    Args:
        token: The token to validate.

    Returns:
        The authenticated account's username.

    Raises:
        AuthenticationError: If the token is invalid or the Hub is unreachable.
    """
    api = HfApi(token=token)
    try:
        info = api.whoami(token=token)
    except HfHubHTTPError as exc:
        raise AuthenticationError(
            message="That HuggingFace token was rejected.",
            suggestions=[
                "Create a fresh token at https://huggingface.co/settings/tokens",
                "A 'Read' token is sufficient.",
            ],
            details=str(exc),
        ) from exc
    except OSError as exc:
        raise AuthenticationError(
            message="Could not reach HuggingFace to validate your token.",
            suggestions=["Check your internet connection.", "See https://status.huggingface.co for outages."],
            details=str(exc),
        ) from exc
    name = info.get("name") if isinstance(info, dict) else None
    return str(name or "your account")


def search_hub_models(query: str, limit: int = 25) -> list[HubModelResult]:
    """Search the Hub for text-generation models matching ``query``.

    Args:
        query: A free-text search string.
        limit: Maximum number of results to return.

    Returns:
        Matching models sorted by the Hub's relevance/popularity. Returns an
        empty list for an empty query.

    Raises:
        ModelNotFoundError: If the Hub cannot be reached.
    """
    if not query.strip():
        return []
    api = HfApi(token=find_hf_token())
    try:
        models = api.list_models(
            search=query,
            task="text-generation",
            sort="downloads",
            direction=-1,
            limit=limit,
        )
    except (HfHubHTTPError, OSError) as exc:
        raise ModelNotFoundError(
            message="Could not search HuggingFace Hub.",
            suggestions=[
                "Check your internet connection.",
                "You can still type a full model id directly, or use a local path.",
            ],
            details=str(exc),
        ) from exc
    results: list[HubModelResult] = []
    for model in models:
        results.append(
            HubModelResult(
                model_id=model.id,
                downloads=getattr(model, "downloads", 0) or 0,
                likes=getattr(model, "likes", 0) or 0,
                gated=bool(getattr(model, "gated", False)),
            )
        )
    return results


def ensure_model_access(model_id: str, token: str | None = None) -> None:
    """Verify the current credentials can access ``model_id``.

    Args:
        model_id: The repository id to check.
        token: An explicit token; falls back to :func:`find_hf_token`.

    Raises:
        AuthenticationError: If the repo is gated and access is not granted.
        ModelNotFoundError: If the repo does not exist or the Hub is unreachable.
    """
    token = token or find_hf_token()
    api = HfApi(token=token)
    try:
        api.model_info(model_id, token=token)
    except GatedRepoError as exc:
        raise AuthenticationError(
            message=f"Access to '{model_id}' has not been granted to your account.",
            suggestions=[
                f"Accept the license at https://huggingface.co/{model_id}",
                "Then ensure your token is set (huggingface-cli login or HF_TOKEN).",
                "It can take a minute after accepting for access to propagate.",
            ],
            details=str(exc),
        ) from exc
    except RepositoryNotFoundError as exc:
        # A private/gated repo also 404s without auth — guide accordingly.
        if token is None:
            raise AuthenticationError(
                message=f"'{model_id}' was not found, or it is gated and needs a token.",
                suggestions=[
                    "If it is gated, log in: huggingface-cli login (or set HF_TOKEN).",
                    f"Otherwise check the id at https://huggingface.co/{model_id}",
                ],
                details=str(exc),
            ) from exc
        raise ModelNotFoundError(
            message=f"Model '{model_id}' does not exist on HuggingFace Hub.",
            suggestions=["Double-check the spelling.", "Browse models at https://huggingface.co/models"],
            details=str(exc),
        ) from exc
    except (HfHubHTTPError, OSError) as exc:
        raise ModelNotFoundError(
            message=f"Could not verify access to '{model_id}'.",
            suggestions=["Check your internet connection.", "See https://status.huggingface.co for outages."],
            details=str(exc),
        ) from exc


def is_local_model_path(model_id: str) -> bool:
    """Return whether ``model_id`` points at an existing local directory.

    Args:
        model_id: A model id or filesystem path.

    Returns:
        True if the string resolves to an existing directory.
    """
    path = Path(model_id).expanduser()
    return path.is_dir()
