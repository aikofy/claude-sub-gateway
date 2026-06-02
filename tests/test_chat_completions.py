"""Non-streaming /chat/completions: shape, translation, usage, finish_reason."""

from __future__ import annotations

import pytest

from .conftest import (
    API_KEY,
    default_usage,
    result_message,
    text_assistant,
)

pytestmark = pytest.mark.anyio

AUTH = {"Authorization": f"Bearer {API_KEY}"}


async def test_basic_completion_shape(client, install_query):
    install_query(
        [text_assistant("Hello there!"), result_message(usage=default_usage(11, 7))]
    )
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "claude-sonnet-4-6"
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello there!"
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


async def test_model_is_echoed_back(client, install_query):
    # Client sends a friendly alias; the response echoes exactly what was sent.
    install_query([text_assistant("hi"), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "gpt-4o"


async def test_alias_resolves_to_real_claude_model(client, install_query):
    calls = install_query([text_assistant("hi"), result_message(usage=default_usage())])
    await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    # gpt-4o -> claude-sonnet-4-6 by the default alias table.
    assert calls[0]["options"].model == "claude-sonnet-4-6"


async def test_default_model_when_omitted(client, install_query):
    calls = install_query([text_assistant("hi"), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 200
    # Echoes the default model, and resolves to a real Claude id.
    assert resp.json()["model"] == "claude-sonnet-4-6"
    assert calls[0]["options"].model == "claude-sonnet-4-6"


async def test_system_message_routed_to_system_prompt(client, install_query):
    calls = install_query([text_assistant("ok"), result_message(usage=default_usage())])
    await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hello"},
            ]
        },
    )
    opts = calls[0]["options"]
    assert opts.system_prompt == "You are a pirate."
    # Single user turn is passed through verbatim as the prompt.
    assert calls[0]["prompt"] == "Hello"
    # Pure text-generation profile.
    assert opts.allowed_tools == []
    assert opts.max_turns == 1


async def test_multi_turn_history_folded_into_prompt(client, install_query):
    calls = install_query([text_assistant("sure"), result_message(usage=default_usage())])
    await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ]
        },
    )
    prompt = calls[0]["prompt"]
    assert "Human: Hi" in prompt
    assert "Assistant: Hello!" in prompt
    assert "Human: How are you?" in prompt


async def test_finish_reason_length_on_max_tokens(client, install_query):
    install_query(
        [
            text_assistant("truncated", stop_reason="max_tokens"),
            result_message(stop_reason="max_tokens", usage=default_usage()),
        ]
    )
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
    )
    assert resp.json()["choices"][0]["finish_reason"] == "length"


async def test_max_tokens_passed_to_backend_env(client, install_query):
    calls = install_query([text_assistant("hi"), result_message(usage=default_usage())])
    await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 123},
    )
    assert calls[0]["options"].env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS") == "123"


async def test_result_text_fallback_when_no_text_block(client, install_query):
    # No AssistantMessage text blocks, only ResultMessage.result.
    install_query([result_message(usage=default_usage(), result="from-result")])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.json()["choices"][0]["message"]["content"] == "from-result"


async def test_bare_path_without_v1_prefix(client, install_query):
    install_query([text_assistant("hi"), result_message(usage=default_usage())])
    resp = await client.post(
        "/chat/completions",
        headers=AUTH,
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi"


async def test_backend_rate_limit_maps_to_429(client, install_query):
    install_query(
        [result_message(is_error=True, subtype="error", api_error_status=429)]
    )
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 429
    assert resp.json()["error"]["type"] == "rate_limit_error"


async def test_missing_messages_returns_openai_400(client):
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "claude-sonnet-4-6"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
