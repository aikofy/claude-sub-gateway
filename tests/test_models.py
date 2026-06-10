"""GET /v1/models: well-formed OpenAI model objects."""

from __future__ import annotations

import pytest

from .conftest import API_KEY

pytestmark = pytest.mark.anyio

AUTH = {"Authorization": f"Bearer {API_KEY}"}


async def test_models_listing_shape(client):
    resp = await client.get("/v1/models", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list) and body["data"]
    for model in body["data"]:
        assert model["object"] == "model"
        assert isinstance(model["id"], str) and model["id"]
        assert isinstance(model["created"], int)
        assert model["owned_by"] == "anthropic"


async def test_models_include_canonical_ids(client):
    resp = await client.get("/v1/models", headers=AUTH)
    ids = {m["id"] for m in resp.json()["data"]}
    assert {
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    } <= ids


async def test_models_include_friendly_aliases(client):
    resp = await client.get("/v1/models", headers=AUTH)
    ids = {m["id"] for m in resp.json()["data"]}
    # A couple of the default aliases should be advertised for model pickers.
    assert "gpt-4o" in ids
    assert "opus" in ids


async def test_models_bare_path(client):
    resp = await client.get("/models", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["object"] == "list"


async def test_models_requires_auth(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 401
