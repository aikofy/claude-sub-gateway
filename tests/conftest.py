"""Shared pytest fixtures and Agent-SDK mocking helpers.

The Claude Agent SDK is mocked everywhere: we monkeypatch the ``query`` symbol
imported into ``app.claude_backend`` so the whole suite runs without a live
subscription or any CLI subprocess. The fake yields *real* SDK message objects
(``AssistantMessage``/``ResultMessage``/``StreamEvent``/``TextBlock``) so the
backend's ``isinstance`` dispatch is exercised faithfully.
"""

from __future__ import annotations

import os
from collections.abc import Callable

# Configure the gateway BEFORE app modules import (settings are read at import).
os.environ.setdefault("GATEWAY_API_KEYS", "test-key-123,second-key")
os.environ.setdefault("DEFAULT_MODEL", "claude-sonnet-4-6")
os.environ.setdefault("REQUEST_TIMEOUT", "30")
os.environ.setdefault("CORS_ORIGINS", "*")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import httpx  # noqa: E402
import pytest  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
)

API_KEY = "test-key-123"
SECOND_KEY = "second-key"


# --------------------------------------------------------------------------- anyio
@pytest.fixture
def anyio_backend() -> str:
    # Pin to asyncio so we don't require trio to be installed.
    return "asyncio"


# ------------------------------------------------------------------- app & client
@pytest.fixture
def app_main():
    from app import config

    config.get_settings.cache_clear()
    from app import main

    # Refresh the cached settings reference on the module under test.
    main.app.state.settings = config.get_settings()
    return main


@pytest.fixture
async def client(app_main):
    fastapi_app = app_main.app
    # Run lifespan manually (ASGITransport does not trigger startup/shutdown).
    async with fastapi_app.router.lifespan_context(fastapi_app):
        transport = ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            yield c


# --------------------------------------------------------- SDK message builders
def text_assistant(
    text: str,
    *,
    model: str = "claude-sonnet-4-6",
    stop_reason: str | None = "end_turn",
    error: str | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        model=model,
        stop_reason=stop_reason,
        error=error,
    )


def result_message(
    *,
    usage: dict | None = None,
    stop_reason: str | None = "end_turn",
    is_error: bool = False,
    result: str | None = None,
    subtype: str = "success",
    api_error_status: int | None = None,
    errors: list | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=12,
        duration_api_ms=10,
        is_error=is_error,
        num_turns=1,
        session_id="test-session",
        stop_reason=stop_reason,
        total_cost_usd=0.0012,
        usage=usage,
        result=result,
        api_error_status=api_error_status,
        errors=errors,
    )


def text_delta_event(text: str) -> StreamEvent:
    return StreamEvent(
        uuid="evt",
        session_id="test-session",
        event={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def default_usage(input_tokens: int = 11, output_tokens: int = 7) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


# ------------------------------------------------------------- query() installer
@pytest.fixture
def install_query(monkeypatch, app_main):
    """Return a function that installs a fake ``query`` and records its calls.

    Usage::

        calls = install_query(lambda prompt, options: [text_assistant("hi"), result_message(...)])
        # ... make request ...
        assert calls[0]["options"].system_prompt == "..."
    """
    calls: list[dict] = []

    def _install(builder: Callable | list):
        async def fake_query(*, prompt, options=None, transport=None):
            calls.append({"prompt": prompt, "options": options})
            messages = builder(prompt, options) if callable(builder) else builder
            for message in messages:
                yield message

        # The backend did `from claude_agent_sdk import query`, so patch the name
        # in the backend's own namespace.
        import app.claude_backend as backend

        monkeypatch.setattr(backend, "query", fake_query)
        return calls

    return _install
