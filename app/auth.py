"""Bearer-key authentication for the gateway.

Clients authenticate exactly like a generic OpenAI/LLM API:

    Authorization: Bearer <KEY>

Keys are matched against ``GATEWAY_API_KEYS`` (comma-separated) using a
constant-time comparison. A missing or invalid key produces an OpenAI-style
401 error body.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Depends, Request

from .config import Settings, get_settings
from .errors import GatewayError

logger = logging.getLogger("claude_gateway.auth")

_UNAUTHORIZED_MESSAGE = (
    "Incorrect API key provided. You can find your key in the gateway "
    "configuration (GATEWAY_API_KEYS). Send it as 'Authorization: Bearer <KEY>'."
)


def _extract_bearer(request: Request) -> str | None:
    """Pull the token out of the Authorization header (Bearer scheme)."""
    header = request.headers.get("authorization") or request.headers.get(
        "Authorization"
    )
    if not header:
        return None
    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _matches_any(token: str, valid_keys: list[str]) -> bool:
    """Constant-time membership test.

    We compare against every configured key (never short-circuiting on the first
    mismatch) so timing does not leak which/how-many keys exist.
    """
    matched = False
    token_bytes = token.encode("utf-8")
    for key in valid_keys:
        if hmac.compare_digest(token_bytes, key.encode("utf-8")):
            matched = True
    return matched


def _unauthorized() -> GatewayError:
    return GatewayError(
        _UNAUTHORIZED_MESSAGE,
        status_code=401,
        type="invalid_request_error",
        code="invalid_api_key",
    )


def require_api_key(
    request: Request, settings: Settings = Depends(get_settings)
) -> str:
    """FastAPI dependency: authenticate the request, returning the matched key.

    Raises :class:`GatewayError` (rendered as an OpenAI 401) on any failure.
    """
    valid_keys = settings.api_keys
    if not valid_keys:
        # Secure-by-default: with no keys configured we reject everything rather
        # than silently allowing unauthenticated access.
        logger.warning(
            "No GATEWAY_API_KEYS configured; rejecting authenticated request. "
            "Set GATEWAY_API_KEYS to enable the gateway."
        )
        raise _unauthorized()

    token = _extract_bearer(request)
    if token is None or not _matches_any(token, valid_keys):
        raise _unauthorized()
    return token
