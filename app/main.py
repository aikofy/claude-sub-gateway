"""FastAPI application: OpenAI-compatible routes over the Claude backend."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from . import __version__
from .auth import require_api_key
from .claude_backend import ClaudeBackend
from .config import Settings, get_settings
from .errors import GatewayError, error_body
from .schemas import (
    ChatCompletion,
    ChatCompletionRequest,
    Model,
    ModelList,
)

logger = logging.getLogger("claude_gateway")

# Stable creation timestamp for advertised models (2024-01-01 UTC).
_MODEL_CREATED = 1704067200


def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("claude_gateway").setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _configure_logging(settings)
    app.state.settings = settings
    app.state.backend = ClaudeBackend(settings)
    if not settings.api_keys:
        logger.warning(
            "GATEWAY_API_KEYS is empty — all requests will be rejected with 401. "
            "Set it (comma-separated) to enable the gateway."
        )
    logger.info(
        "Claude Subscription Gateway v%s ready (default_model=%s, max_concurrency=%s)",
        __version__,
        settings.default_model,
        settings.max_concurrency,
    )
    yield


app = FastAPI(
    title="Claude Subscription Gateway",
    version=__version__,
    description=(
        "OpenAI-compatible API backed by a Claude subscription via the "
        "Claude Agent SDK."
    ),
    lifespan=lifespan,
)

# Permissive (configurable) CORS so browser clients (e.g. OpenWebUI) can call in.
_cors_origins = get_settings().cors_origin_list
# If "*" appears anywhere in the list, treat it as a pure wildcard: Starlette then
# reflects any Origin, which must NOT be combined with credentials (that would let
# any site make credentialed cross-origin calls). The gateway uses header Bearer
# auth (no cookies), so disabling credentials under wildcard loses nothing.
_cors_wildcard = "*" in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_wildcard else _cors_origins,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- helpers
def _get_backend(request: Request) -> ClaudeBackend:
    return request.app.state.backend


def _sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def _sse_event_stream(
    backend: ClaudeBackend, req: ChatCompletionRequest
) -> AsyncIterator[str]:
    """Serialize backend chunks as OpenAI SSE; always terminate with [DONE].

    Each event is ``data: {json}\\n\\n``. On a mid-stream error we emit an OpenAI
    error object as a data event so clients can surface it, then close cleanly.
    """
    try:
        async for chunk in backend.stream_chunks(req):
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected / server shutting down — stop without emitting more.
        raise
    except GatewayError as exc:
        logger.info("Stream ended with error: %s", exc.message)
        yield _sse_data(exc.to_body())
    except Exception:  # noqa: BLE001 - never leak internals/traceback to the wire
        logger.exception("Unexpected error during streaming")
        yield _sse_data(error_body("Internal server error.", type="api_error"))
    yield "data: [DONE]\n\n"


async def _complete_or_disconnect(
    request: Request, backend: ClaudeBackend, body: ChatCompletionRequest
) -> ChatCompletion:
    """Run a non-streaming completion, cancelling the in-flight query if the
    client disconnects.

    Unlike streaming responses, a regular endpoint coroutine is NOT auto-cancelled
    by Starlette/uvicorn when the client goes away — uvicorn just posts
    ``http.disconnect`` on the receive channel. We race the generation against a
    disconnect poll so an abandoned request frees its concurrency slot and tears
    down the CLI subprocess instead of running to completion / the timeout.
    """
    task: asyncio.Task[ChatCompletion] = asyncio.ensure_future(backend.complete(body))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.5)
            if task in done:
                return task.result()
            if await request.is_disconnected():
                logger.info("Client disconnected; cancelling in-flight completion.")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                # Nothing will receive this; abort the request cleanly.
                raise ClientDisconnect()
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


# --------------------------------------------------------------------------- routes
api_router = APIRouter()


@api_router.get("/models", response_model=ModelList)
async def list_models(
    request: Request, _: str = Depends(require_api_key)
) -> ModelList:
    settings: Settings = request.app.state.settings
    data = [
        Model(id=name, created=_MODEL_CREATED, owned_by="anthropic")
        for name in settings.advertised_models()
    ]
    return ModelList(data=data)


@api_router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    _: str = Depends(require_api_key),
):
    backend = _get_backend(request)

    if body.stream:
        return StreamingResponse(
            _sse_event_stream(backend, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Disable proxy buffering so chunks flush immediately.
                "X-Accel-Buffering": "no",
            },
        )

    completion: ChatCompletion = await _complete_or_disconnect(request, backend, body)
    return completion


# Mount the OpenAI surface under both /v1 (canonical) and the bare prefix
# (some clients omit /v1).
app.include_router(api_router, prefix="/v1")
app.include_router(api_router, prefix="")


@app.get("/health")
async def health() -> dict:
    """Liveness probe — no auth."""
    return {"status": "ok", "version": __version__}


@app.get("/")
async def root() -> dict:
    return {
        "service": "claude-subscription-gateway",
        "version": __version__,
        "docs": "/docs",
        "openai_base_url": "/v1",
    }


# --------------------------------------------------------------- exception handlers
@app.exception_handler(GatewayError)
async def _handle_gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
    headers = {}
    if exc.status_code == 401:
        # Standard challenge header for 401s.
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=exc.status_code, content=exc.to_body(), headers=headers
    )


@app.exception_handler(RequestValidationError)
async def _handle_validation_error(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render request-validation problems as an OpenAI 400 error (not 422)."""
    try:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        message = first.get("msg", "Invalid request.")
        if loc:
            message = f"{message} (at '{loc}')"
        param = loc or None
    except (IndexError, KeyError, TypeError):
        message, param = "Invalid request.", None
    return JSONResponse(
        status_code=400,
        content=error_body(
            message, type="invalid_request_error", code="invalid_request", param=param
        ),
    )


@app.exception_handler(ClientDisconnect)
async def _handle_client_disconnect(_: Request, __: ClientDisconnect) -> JSONResponse:
    # The client is already gone; this response is not delivered. Returning a
    # clean status (rather than a 500) keeps the logs honest.
    return JSONResponse(
        status_code=499,
        content=error_body(
            "Client disconnected.", type="api_error", code="client_disconnect"
        ),
    )


@app.exception_handler(Exception)
async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    # Log the real error server-side; never echo internals/traceback on the wire.
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content=error_body("Internal server error.", type="api_error"),
    )
