"""Streaming /chat/completions: SSE shape and incremental deltas."""

from __future__ import annotations

import json

import pytest

from .conftest import (
    API_KEY,
    default_usage,
    result_message,
    text_assistant,
    text_delta_event,
)

pytestmark = pytest.mark.anyio

AUTH = {"Authorization": f"Bearer {API_KEY}"}


def _parse_sse(raw: str) -> list:
    """Return parsed JSON payloads from an SSE body (excluding the [DONE] marker)."""
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(data))
    return events


async def _collect_stream(client, payload) -> str:
    body = ""
    async with client.stream(
        "POST", "/v1/chat/completions", headers=AUTH, json=payload
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for chunk in resp.aiter_text():
            body += chunk
    return body


async def test_streaming_basic_shape(client, install_query):
    install_query(
        [
            text_delta_event("Hello"),
            text_delta_event(", "),
            text_delta_event("world!"),
            text_assistant("Hello, world!"),
            result_message(usage=default_usage()),
        ]
    )
    body = await _collect_stream(
        client,
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )
    events = _parse_sse(body)

    # Terminates with [DONE].
    assert events[-1] == "[DONE]"
    json_events = [e for e in events if e != "[DONE]"]

    # First chunk announces the assistant role.
    first = json_events[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"].get("role") == "assistant"

    # Concatenated content deltas reconstruct the full message.
    text = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in json_events
    )
    assert text == "Hello, world!"

    # Some chunk carries the terminal finish_reason.
    finish = [
        e["choices"][0]["finish_reason"]
        for e in json_events
        if e["choices"] and e["choices"][0].get("finish_reason")
    ]
    assert finish[-1] == "stop"

    # Every chunk echoes the requested model.
    for e in json_events:
        assert e["model"] == "claude-sonnet-4-6"


async def test_streaming_is_incremental_not_single_blob(client, install_query):
    install_query(
        [
            text_delta_event("A"),
            text_delta_event("B"),
            text_delta_event("C"),
            text_assistant("ABC"),
            result_message(usage=default_usage()),
        ]
    )
    body = await _collect_stream(
        client, {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
    )
    json_events = [e for e in _parse_sse(body) if e != "[DONE]"]
    content_chunks = [
        e for e in json_events if e["choices"] and e["choices"][0]["delta"].get("content")
    ]
    # Three separate text deltas => at least three content chunks (not buffered).
    assert len(content_chunks) >= 3


async def test_streaming_usage_chunk_when_requested(client, install_query):
    install_query(
        [
            text_delta_event("hi"),
            text_assistant("hi"),
            result_message(usage=default_usage(20, 5)),
        ]
    )
    body = await _collect_stream(
        client,
        {
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    json_events = [e for e in _parse_sse(body) if e != "[DONE]"]
    usage_events = [e for e in json_events if e.get("usage")]
    assert usage_events, "expected a usage chunk when include_usage=true"
    usage = usage_events[-1]["usage"]
    assert usage["prompt_tokens"] == 20
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    # OpenAI's usage chunk has an empty choices list.
    assert usage_events[-1]["choices"] == []


async def test_streaming_no_usage_chunk_by_default(client, install_query):
    install_query(
        [text_delta_event("hi"), text_assistant("hi"), result_message(usage=default_usage())]
    )
    body = await _collect_stream(
        client, {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
    )
    json_events = [e for e in _parse_sse(body) if e != "[DONE]"]
    assert not any(e.get("usage") for e in json_events)


async def test_streaming_fallback_without_partial_events(client, install_query):
    # No StreamEvents at all (older CLI); backend should still stream the text.
    install_query([text_assistant("fallback text"), result_message(usage=default_usage())])
    body = await _collect_stream(
        client, {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
    )
    json_events = [e for e in _parse_sse(body) if e != "[DONE]"]
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in json_events)
    assert text == "fallback text"


async def test_streaming_error_emits_error_event_then_done(client, install_query):
    install_query(
        [result_message(is_error=True, subtype="error", api_error_status=500)]
    )
    body = await _collect_stream(
        client, {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
    )
    events = _parse_sse(body)
    assert events[-1] == "[DONE]"
    error_events = [e for e in events if e != "[DONE]" and "error" in e]
    assert error_events, "expected an inline error event"
