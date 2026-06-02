"""Gateway error type and OpenAI-style error envelope helpers."""

from __future__ import annotations

from typing import Any

from .schemas import ErrorDetail, ErrorResponse


class GatewayError(Exception):
    """An error that should be rendered as an OpenAI-style error response.

    Carries everything needed to build both the HTTP status and the
    ``{"error": {...}}`` body that OpenAI clients expect.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        type: str = "api_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = type
        self.code = code
        self.param = param

    def to_body(self) -> dict[str, Any]:
        return error_body(
            self.message, type=self.type, code=self.code, param=self.param
        )


def error_body(
    message: str,
    *,
    type: str = "api_error",
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    """Build the OpenAI error envelope ``{"error": {...}}`` as a plain dict."""
    return ErrorResponse(
        error=ErrorDetail(message=message, type=type, code=code, param=param)
    ).model_dump()
