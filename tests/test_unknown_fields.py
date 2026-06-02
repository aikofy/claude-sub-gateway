"""Tolerance for unknown / client-specific fields and content shapes."""

from __future__ import annotations

import pytest

from .conftest import API_KEY, default_usage, result_message, text_assistant

pytestmark = pytest.mark.anyio

AUTH = {"Authorization": f"Bearer {API_KEY}"}


async def test_unknown_top_level_fields_are_ignored(client, install_query):
    install_query([text_assistant("ok"), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hi"}],
            # A grab-bag of params real clients send that we don't use.
            "n": 1,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.2,
            "logit_bias": {"50256": -100},
            "user": "user-123",
            "seed": 42,
            "response_format": {"type": "json_object"},
            "tools": [{"type": "function", "function": {"name": "x"}}],
            "tool_choice": "auto",
            "logprobs": True,
            "top_logprobs": 5,
            "parallel_tool_calls": True,
            "service_tier": "auto",
            "some_totally_made_up_field": {"deep": [1, 2, 3]},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"


async def test_unknown_fields_in_message_are_ignored(client, install_query):
    install_query([text_assistant("ok"), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hi",
                    "name": "alice",
                    "weird_extra": True,
                }
            ]
        },
    )
    assert resp.status_code == 200


async def test_content_as_parts_array_is_flattened(client, install_query):
    calls = install_query([text_assistant("ok"), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ]
        },
    )
    assert resp.status_code == 200
    # Text parts joined; non-text (image) part ignored.
    assert "part one" in calls[0]["prompt"]
    assert "part two" in calls[0]["prompt"]


async def test_stop_as_list_is_accepted(client, install_query):
    install_query([text_assistant("ok"), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["\n\n", "END"],
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )
    assert resp.status_code == 200


async def test_null_content_assistant_message_is_tolerated(client, install_query):
    # Assistant tool-call turns can carry content: null.
    install_query([text_assistant("ok"), result_message(usage=default_usage())])
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": None},
                {"role": "user", "content": "Still there?"},
            ]
        },
    )
    assert resp.status_code == 200
