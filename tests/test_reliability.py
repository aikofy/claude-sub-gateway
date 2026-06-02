"""Reliability behaviors: per-request timeout, concurrency cap, and teardown /
client-disconnect cancellation.

These drive the backend (and the non-stream disconnect helper) directly with
custom fake ``query`` generators so the async behavior is deterministic — the
HTTP-level tests can't exercise stalls / disconnects reliably.
"""

from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk import (
    CLIConnectionError,
    CLINotFoundError,
    ProcessError,
)
from starlette.requests import ClientDisconnect

from app.claude_backend import ClaudeBackend
from app.config import Settings
from app.errors import GatewayError
from app.main import _complete_or_disconnect
from app.schemas import ChatCompletionRequest

from .conftest import API_KEY, default_usage, result_message, text_delta_event

pytestmark = pytest.mark.anyio


def make_backend(**overrides) -> ClaudeBackend:
    settings = Settings(gateway_api_keys="k", **overrides)
    return ClaudeBackend(settings)


def make_req(stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hi"}], stream=stream
    )


# --------------------------------------------------------------------- max_turns
def test_max_turns_defaults_above_one_and_is_configurable():
    # Default must be > 1 (max_turns=1 makes some models, e.g. Haiku, abort with
    # "Reached maximum number of turns (1)" -> 502).
    default_opts = make_backend()._build_options(
        system_prompt=None, model="m", max_tokens=None, stream=False
    )
    assert default_opts.max_turns > 1

    custom = make_backend(max_turns=5)._build_options(
        system_prompt=None, model="m", max_tokens=None, stream=False
    )
    assert custom.max_turns == 5

    # Never drops below 1 even if misconfigured to 0/negative.
    clamped = make_backend(max_turns=0)._build_options(
        system_prompt=None, model="m", max_tokens=None, stream=False
    )
    assert clamped.max_turns == 1


# --------------------------------------------------------------------- timeout
async def test_nonstream_timeout_raises_504(monkeypatch):
    async def slow_query(*, prompt, options=None, transport=None):
        await asyncio.sleep(5)
        yield result_message(usage=default_usage())  # never reached

    monkeypatch.setattr("app.claude_backend.query", slow_query)
    backend = make_backend(request_timeout=0.15)

    with pytest.raises(GatewayError) as ei:
        await backend.complete(make_req())
    assert ei.value.status_code == 504
    assert ei.value.code == "timeout"


async def test_stream_timeout_raises_504_into_consumer(monkeypatch):
    # The deadline must reliably raise into the consumer even though the first
    # chunk was already yielded (the key streaming-timeout fix).
    async def stalling_query(*, prompt, options=None, transport=None):
        yield text_delta_event("Hi")
        await asyncio.sleep(5)
        yield result_message(usage=default_usage())  # never reached

    monkeypatch.setattr("app.claude_backend.query", stalling_query)
    backend = make_backend(request_timeout=0.15)

    got_content = False
    with pytest.raises(GatewayError) as ei:
        async for chunk in backend.stream_chunks(make_req(stream=True)):
            if chunk.choices and chunk.choices[0].delta.content:
                got_content = True
    assert got_content  # streamed before the stall
    assert ei.value.status_code == 504
    assert ei.value.code == "timeout"


async def test_slow_consumer_does_not_trip_timeout(monkeypatch):
    # A slow *client* (slow to read) must NOT count against the idle timeout.
    async def trickle_query(*, prompt, options=None, transport=None):
        for _ in range(3):
            yield text_delta_event("x")
        yield result_message(usage=default_usage())

    monkeypatch.setattr("app.claude_backend.query", trickle_query)
    backend = make_backend(request_timeout=0.2)

    pieces = 0
    async for chunk in backend.stream_chunks(make_req(stream=True)):
        if chunk.choices and chunk.choices[0].delta.content:
            pieces += 1
            await asyncio.sleep(0.15)  # consumer slower than the idle timeout
    # No timeout raised despite total time > request_timeout; all content arrived.
    assert pieces == 3


# ----------------------------------------------------------------- concurrency
async def test_concurrency_is_capped(monkeypatch):
    active = 0
    max_seen = 0
    release = asyncio.Event()

    async def blocking_query(*, prompt, options=None, transport=None):
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        try:
            await release.wait()
            yield result_message(usage=default_usage(), result="ok")
        finally:
            active -= 1

    monkeypatch.setattr("app.claude_backend.query", blocking_query)
    backend = make_backend(max_concurrency=2, request_timeout=30)

    tasks = [asyncio.ensure_future(backend.complete(make_req())) for _ in range(5)]
    await asyncio.sleep(0.1)  # let them all try to enter
    assert max_seen == 2  # never more than the cap ran at once
    assert active == 2

    release.set()
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    assert all(r.choices[0].message.content == "ok" for r in results)


# --------------------------------------------------- teardown / disconnect
async def test_closing_stream_tears_down_query(monkeypatch):
    torn_down = asyncio.Event()

    async def gen_query(*, prompt, options=None, transport=None):
        try:
            yield text_delta_event("A")
            yield text_delta_event("B")
            await asyncio.sleep(10)
        finally:
            torn_down.set()

    monkeypatch.setattr("app.claude_backend.query", gen_query)
    backend = make_backend(request_timeout=30)

    agen = backend.stream_chunks(make_req(stream=True))
    await agen.__anext__()  # initial role chunk
    await agen.__anext__()  # first content chunk
    await agen.aclose()  # simulate client disconnect / stream close

    assert torn_down.is_set()


async def test_nonstream_disconnect_cancels_query(monkeypatch):
    started = asyncio.Event()
    torn_down = asyncio.Event()

    async def blocking_query(*, prompt, options=None, transport=None):
        started.set()
        try:
            await asyncio.sleep(10)
            yield result_message(usage=default_usage())  # never reached
        finally:
            torn_down.set()

    monkeypatch.setattr("app.claude_backend.query", blocking_query)
    backend = make_backend(request_timeout=30)

    class FakeRequest:
        async def is_disconnected(self) -> bool:
            return started.is_set()

    with pytest.raises(ClientDisconnect):
        await _complete_or_disconnect(FakeRequest(), backend, make_req())

    assert torn_down.is_set()  # the in-flight query was cancelled & cleaned up


# ------------------------------------------------- SDK exception -> error mapping
@pytest.mark.parametrize(
    "exc, status, code",
    [
        (ProcessError("died", exit_code=1, stderr="/home/secret/.claude/creds leaked"),
         502, "cli_process_error"),
        (CLINotFoundError("Claude Code not found", cli_path="/usr/bin/claude"),
         500, "cli_not_found"),
        (CLIConnectionError("connection refused to 127.0.0.1"),
         502, "cli_connection_error"),
    ],
)
async def test_sdk_exception_mapping_and_no_leak(monkeypatch, exc, status, code):
    async def raising_query(*, prompt, options=None, transport=None):
        raise exc
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("app.claude_backend.query", raising_query)
    backend = make_backend(request_timeout=30)

    with pytest.raises(GatewayError) as ei:
        await backend.complete(make_req())
    err = ei.value
    assert err.status_code == status
    assert err.code == code
    # The client-facing message must NOT echo raw exception text / stderr / paths.
    assert "secret" not in err.message.lower()
    assert "stderr" not in err.message.lower()
    assert str(exc) not in err.message


async def test_http_returns_499_on_client_disconnect(client, monkeypatch):
    # End-to-end: a disconnect during a non-streaming request yields a 499 with the
    # OpenAI error envelope (verifies the specific handler wins over the generic one).
    async def slow_query(*, prompt, options=None, transport=None):
        await asyncio.sleep(5)
        yield result_message(usage=default_usage())  # never reached

    monkeypatch.setattr("app.claude_backend.query", slow_query)

    async def _disconnected(self) -> bool:
        return True

    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", _disconnected
    )

    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 499
    assert resp.json()["error"]["code"] == "client_disconnect"
