"""End-to-end check with the *official* ``openai`` client.

Spins up a real uvicorn server (so HTTP/SSE goes over a socket), with the Agent
SDK's ``query`` mocked, and drives it with ``openai.OpenAI`` exactly as a real
user would. Directly exercises the acceptance criteria. Skipped if ``openai``
isn't installed.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

openai = pytest.importorskip("openai")

import uvicorn  # noqa: E402
from unittest.mock import patch  # noqa: E402

from .conftest import (  # noqa: E402
    API_KEY,
    default_usage,
    result_message,
    text_assistant,
    text_delta_event,
)

EXPECTED = "Hello from Claude!"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ThreadedServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # not on the main thread
        pass


async def _fake_query(*, prompt, options=None, transport=None):
    for message in (
        text_delta_event("Hello "),
        text_delta_event("from "),
        text_delta_event("Claude!"),
        text_assistant(EXPECTED),
        result_message(usage=default_usage(13, 4)),
    ):
        yield message


@pytest.fixture
def live_base_url():
    import app.claude_backend as backend
    import app.config as config

    config.get_settings.cache_clear()
    import app.main as main

    port = _free_port()
    cfg = uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning")
    server = _ThreadedServer(cfg)
    thread = threading.Thread(target=server.run, daemon=True)

    with patch.object(backend, "query", _fake_query):
        thread.start()
        deadline = time.time() + 15
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started, "uvicorn server failed to start"
        try:
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            server.should_exit = True
            thread.join(timeout=15)


def _client(base_url: str, key: str = API_KEY) -> "openai.OpenAI":
    return openai.OpenAI(base_url=base_url, api_key=key, max_retries=0)


def test_openai_nonstreaming(live_base_url):
    client = _client(live_base_url)
    resp = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
    )
    assert resp.choices[0].message.content == EXPECTED
    assert resp.choices[0].finish_reason == "stop"
    assert resp.model == "gpt-4o"  # echoes the requested model
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens
    assert resp.usage.total_tokens > 0


def test_openai_streaming(live_base_url):
    client = _client(live_base_url)
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    pieces = []
    finish = None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            pieces.append(delta.content)
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    assert "".join(pieces) == EXPECTED
    assert finish == "stop"


def test_openai_streaming_with_usage(live_base_url):
    client = _client(live_base_url)
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    usage = None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
    assert usage is not None
    assert usage.total_tokens > 0


def test_openai_models_list(live_base_url):
    client = _client(live_base_url)
    ids = {m.id for m in client.models.list().data}
    assert "claude-sonnet-4-6" in ids
    assert "claude-opus-4-8" in ids


def test_openai_invalid_key_raises_auth_error(live_base_url):
    client = _client(live_base_url, key="totally-wrong")
    with pytest.raises(openai.AuthenticationError):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )
