"""Function/tool calling: prompt injection, tool-call parsing, the round-trip,
and OpenAI-shaped responses (non-streaming + streaming).

These exercise both the HTTP surface (via the ``client`` fixture) and the parser
/ prompt-builder directly, since the latter has many tolerant-parsing branches.
"""

from __future__ import annotations

import json

import pytest

from app.claude_backend import _coerce_arguments, _parse_tool_calls

from .conftest import (
    API_KEY,
    default_usage,
    result_message,
    text_assistant,
    text_delta_event,
)

pytestmark = pytest.mark.anyio

AUTH = {"Authorization": f"Bearer {API_KEY}"}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def _tool_call_json(name: str, arguments: dict) -> str:
    return json.dumps({"tool_calls": [{"name": name, "arguments": arguments}]})


# =============================================================================
# Parser unit tests (tolerant extraction)
# =============================================================================
@pytest.mark.parametrize(
    "text, expected",
    [
        # canonical wrapper
        ('{"tool_calls":[{"name":"f","arguments":{"a":1}}]}', [("f", '{"a":1}')]),
        # single bare object
        ('{"name":"f","arguments":{"a":1}}', [("f", '{"a":1}')]),
        # bare array
        ('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',
         [("a", "{}"), ("b", "{}")]),
        # fenced code block
        ('```json\n{"name":"f","arguments":{"q":"x, y"}}\n```', [("f", '{"q":"x, y"}')]),
        # leading prose + braces inside string literals must not break balance
        ('Sure: {"tool_calls":[{"name":"f","arguments":{"s":"}{["}}]}',
         [("f", '{"s":"}{["}')]),
        # arguments already a JSON string -> kept as JSON string
        ('{"name":"f","arguments":"{\\"k\\":true}"}', [("f", '{"k":true}')]),
        # arguments omitted -> defaults to {}
        ('{"name":"f"}', [("f", "{}")]),
        # plain text is not a tool call
        ("I cannot help with that.", None),
        # empty
        ("", None),
        # JSON that isn't a tool-call shape
        ('{"foo": "bar"}', None),
    ],
)
def test_parse_tool_calls(text, expected):
    parsed = _parse_tool_calls(text)
    if expected is None:
        assert parsed is None
    else:
        assert [(c.name, c.arguments) for c in parsed] == expected


def test_parse_coerces_object_arguments_to_json_string():
    parsed = _parse_tool_calls('{"name":"f","arguments":{"n":1,"b":true}}')
    assert parsed is not None
    # arguments must be a *string* (OpenAI shape), and valid JSON.
    assert isinstance(parsed[0].arguments, str)
    assert json.loads(parsed[0].arguments) == {"n": 1, "b": True}


# =============================================================================
# Prompt injection
# =============================================================================
async def test_tools_injected_into_system_prompt(client, install_query):
    calls = install_query([text_assistant("hi"), result_message(usage=default_usage())])
    await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "weather?"},
            ],
            "tools": [WEATHER_TOOL],
        },
    )
    sysprompt = calls[0]["options"].system_prompt
    assert "Be terse." in sysprompt  # original system content preserved
    assert "get_weather" in sysprompt  # tool name present
    assert "tool_calls" in sysprompt  # protocol present


async def test_tool_choice_none_does_not_inject(client, install_query):
    calls = install_query([text_assistant("hi"), result_message(usage=default_usage())])
    await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        },
    )
    # No tool instructions when tool_choice == "none".
    assert calls[0]["options"].system_prompt is None


async def test_tool_choice_required_instructs_must_call(client, install_query):
    calls = install_query(
        [text_assistant(_tool_call_json("get_weather", {"location": "Paris"})),
         result_message(usage=default_usage())]
    )
    await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
        },
    )
    assert "MUST call at least one" in calls[0]["options"].system_prompt


async def test_tool_choice_named_function_instructs_specific_tool(client, install_query):
    calls = install_query(
        [text_assistant(_tool_call_json("get_weather", {"location": "Paris"})),
         result_message(usage=default_usage())]
    )
    await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
    )
    assert 'MUST call the tool "get_weather"' in calls[0]["options"].system_prompt


async def test_legacy_functions_and_function_call(client, install_query):
    calls = install_query(
        [text_assistant(_tool_call_json("get_weather", {"location": "Paris"})),
         result_message(usage=default_usage())]
    )
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "functions": [WEATHER_TOOL["function"]],
            "function_call": "auto",
        },
    )
    assert "get_weather" in calls[0]["options"].system_prompt
    assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"


# =============================================================================
# Non-streaming responses
# =============================================================================
async def test_nonstream_tool_call_response_shape(client, install_query):
    install_query(
        [text_assistant(_tool_call_json("get_weather", {"location": "Paris"})),
         result_message(usage=default_usage())]
    )
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [WEATHER_TOOL],
        },
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["id"].startswith("call_")
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"location": "Paris"}


async def test_nonstream_multiple_tool_calls(client, install_query):
    payload = json.dumps(
        {"tool_calls": [
            {"name": "get_weather", "arguments": {"location": "Paris"}},
            {"name": "get_weather", "arguments": {"location": "Rome"}},
        ]}
    )
    install_query([text_assistant(payload), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather in Paris and Rome?"}],
            "tools": [WEATHER_TOOL],
        },
    )
    calls = resp.json()["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 2
    assert {json.loads(c["function"]["arguments"])["location"] for c in calls} == {
        "Paris",
        "Rome",
    }
    # Each call gets a distinct id.
    assert calls[0]["id"] != calls[1]["id"]


async def test_nonstream_plain_text_despite_tools(client, install_query):
    # Model chooses NOT to call a tool -> normal text content, finish "stop".
    install_query([text_assistant("It is sunny."), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
        },
    )
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "It is sunny."
    assert choice["message"].get("tool_calls") is None


# =============================================================================
# The tool round-trip (client sends results back)
# =============================================================================
async def test_round_trip_tool_result_folded_into_prompt(client, install_query):
    calls = install_query([text_assistant("It's 21°C in Paris."),
                           result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Paris"}',
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": '{"temp_c":21}'},
            ],
            "tools": [WEATHER_TOOL],
        },
    )
    assert resp.status_code == 200
    prompt = calls[0]["prompt"]
    assert "weather in Paris?" in prompt
    assert "Called tool get_weather" in prompt
    # The tool result is labeled with the resolved tool name (from call_abc).
    assert "Tool result (get_weather)" in prompt
    assert '{"temp_c":21}' in prompt


# =============================================================================
# Streaming
# =============================================================================
async def test_stream_tool_call_emits_tool_call_deltas(client, install_query):
    # Model streams a tool-call JSON in fragments; gateway buffers + emits a
    # tool_calls delta, then a finish_reason="tool_calls" chunk.
    install_query([
        text_delta_event('{"tool_calls":[{"name":'),
        text_delta_event('"get_weather","arguments"'),
        text_delta_event(':{"location":"Paris"}}]}'),
        result_message(usage=default_usage()),
    ])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    chunks = _parse_sse(resp.text)
    # The text fragments must NOT be streamed as content (would corrupt clients).
    assert all(
        c["choices"][0]["delta"].get("content") in (None, "")
        for c in chunks
        if c["choices"]
    )
    tool_deltas = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_deltas[0]["function"]["arguments"]) == {"location": "Paris"}
    # Final content chunk carries the tool_calls finish reason.
    finishes = [
        c["choices"][0].get("finish_reason") for c in chunks
        if c["choices"] and c["choices"][0].get("finish_reason")
    ]
    assert finishes == ["tool_calls"]


async def test_stream_plain_text_with_tools_offered(client, install_query):
    # Tools offered but model returns plain text -> buffered, emitted as content,
    # finish "stop".
    install_query([
        text_delta_event("It "),
        text_delta_event("is sunny."),
        result_message(usage=default_usage()),
    ])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": True,
        },
    )
    chunks = _parse_sse(resp.text)
    content = "".join(
        c["choices"][0]["delta"].get("content") or ""
        for c in chunks
        if c["choices"]
    )
    assert content == "It is sunny."
    finishes = [
        c["choices"][0].get("finish_reason") for c in chunks
        if c["choices"] and c["choices"][0].get("finish_reason")
    ]
    assert finishes == ["stop"]
    # No tool_calls anywhere.
    assert not any(
        c["choices"] and c["choices"][0]["delta"].get("tool_calls") for c in chunks
    )


def _parse_sse(text: str) -> list[dict]:
    """Parse an SSE body into the list of JSON data events (excluding [DONE])."""
    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


# =============================================================================
# Regression tests for the adversarial-review findings
# =============================================================================
def test_real_call_not_shadowed_by_fenced_example():
    # An example (non-tool) JSON in a ```json fence must not hide the real call.
    text = (
        "Here's the schema example:\n```json\n{\"foo\": \"bar\"}\n```\n"
        'Now calling: {"tool_calls":[{"name":"get_weather",'
        '"arguments":{"location":"Paris"}}]}'
    )
    parsed = _parse_tool_calls(text)
    assert parsed is not None
    assert [(c.name, c.arguments) for c in parsed] == [
        ("get_weather", '{"location":"Paris"}')
    ]


def test_real_call_not_shadowed_by_stray_object():
    # A stray non-tool object earlier in the (unfenced) reply must not win.
    text = 'config is {"x":1} and the call is {"name":"f","arguments":{"a":2}}'
    parsed = _parse_tool_calls(text)
    assert parsed is not None
    assert [(c.name, c.arguments) for c in parsed] == [("f", '{"a":2}')]


@pytest.mark.parametrize(
    "value",
    [5, [1, 2], "hello", "5", "[1, 2]", '"x"', True, None],
)
def test_coerce_arguments_always_decodes_to_object(value):
    # OpenAI requires function.arguments to be a JSON string encoding an OBJECT.
    out = _coerce_arguments(value)
    assert isinstance(json.loads(out), dict)


def test_coerce_arguments_preserves_real_objects():
    assert json.loads(_coerce_arguments({"n": 1, "b": True})) == {"n": 1, "b": True}
    assert _coerce_arguments({}) == "{}"
    assert _coerce_arguments(None) == "{}"


async def test_legacy_function_call_echo_rendered_in_prompt(client, install_query):
    # A legacy client echoes the assistant call as singular `function_call`; it
    # must still appear in the transcript so the model knows what it called.
    calls = install_query(
        [text_assistant("It's 21°C."), result_message(usage=default_usage())]
    )
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_weather",
                        "arguments": '{"location":"Paris"}',
                    },
                },
                {"role": "function", "name": "get_weather", "content": '{"temp_c":21}'},
            ],
            "functions": [WEATHER_TOOL["function"]],
        },
    )
    assert resp.status_code == 200
    prompt = calls[0]["prompt"]
    assert "Called tool get_weather" in prompt
    assert "Tool result (get_weather)" in prompt


def test_echoed_null_arguments_render_as_empty_object():
    from app.claude_backend import ClaudeBackend
    from app.config import Settings
    from app.schemas import ChatMessage

    be = ClaudeBackend(Settings(gateway_api_keys="k"))
    convo = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": None},
            }],
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="ok"),
    ]
    prompt = be._build_prompt(convo)
    assert "[Called tool f with arguments {}]" in prompt
    assert "arguments null" not in prompt


async def test_stream_tool_call_role_chunk_content_is_null(client, install_query):
    # For a tool-call stream, the role-announce chunk carries content:null (which
    # serializes as an absent key under exclude_none), not "".
    install_query([
        text_delta_event('{"tool_calls":[{"name":"get_weather",'),
        text_delta_event('"arguments":{"location":"Paris"}}]}'),
        result_message(usage=default_usage()),
    ])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": True,
        },
    )
    chunks = _parse_sse(resp.text)
    first_delta = chunks[0]["choices"][0]["delta"]
    assert first_delta.get("role") == "assistant"
    # content must NOT be the empty string for a tool stream (null -> omitted).
    assert first_delta.get("content") is None
