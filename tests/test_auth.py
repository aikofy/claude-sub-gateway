"""Authentication behavior: pass/fail and OpenAI-style 401 envelope."""

from __future__ import annotations

import pytest

from .conftest import API_KEY, SECOND_KEY, result_message, text_assistant

pytestmark = pytest.mark.anyio


async def test_health_requires_no_auth(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_missing_key_returns_401(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 401
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_api_key"
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


async def test_invalid_key_returns_401(client):
    resp = await client.get(
        "/v1/models", headers={"Authorization": "Bearer wrong-key"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


async def test_malformed_auth_header_returns_401(client):
    # No "Bearer " scheme prefix.
    resp = await client.get("/v1/models", headers={"Authorization": API_KEY})
    assert resp.status_code == 401


async def test_valid_key_allows_access(client):
    resp = await client.get(
        "/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}
    )
    assert resp.status_code == 200


async def test_second_configured_key_also_works(client):
    resp = await client.get(
        "/v1/models", headers={"Authorization": f"Bearer {SECOND_KEY}"}
    )
    assert resp.status_code == 200


async def test_auth_precedes_body_validation(client, install_query):
    # Bad key AND invalid body -> auth (401) should win over validation (400).
    install_query([text_assistant("hi"), result_message(usage=None)])
    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer nope"},
        json={"not": "valid"},
    )
    assert resp.status_code == 401
