"""Translation layer between the OpenAI API and the Claude Agent SDK.

The gateway runs a *pure text-generation profile*: every request becomes a
single-turn, tool-less ``query()`` whose authentication is inherited from the
Claude Code CLI's subscription login. Nothing here ever touches an Anthropic API
key or hits ``api.anthropic.com`` directly — all model access flows through the
Agent SDK, which drives the locally installed (and logged-in) CLI.

Public surface
--------------
``ClaudeBackend.complete(req)``        -> a :class:`ChatCompletion`
``ClaudeBackend.stream_chunks(req)``   -> async iterator of :class:`ChatCompletionChunk`
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    ResultMessage,
    StreamEvent,
    TextBlock,
    query,
)

from .config import Settings
from .errors import GatewayError
from .schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    Choice,
    ChunkChoice,
    DeltaMessage,
    FunctionCall,
    FunctionCallDelta,
    ResponseMessage,
    Tool,
    ToolCall,
    ToolCallDelta,
    Usage,
)

logger = logging.getLogger("claude_gateway.backend")

# Upper bound on how long we *wait* for the CLI subprocess to tear down before
# freeing the concurrency slot. The SDK's own close() can take up to ~10s if the
# subprocess ignores stdin-EOF and SIGTERM; we don't hold the slot that long. The
# teardown is shielded, so it still completes (kills the process) in the
# background even if we stop awaiting it.
_TEARDOWN_TIMEOUT = 3.0

# Map Anthropic stop_reason -> OpenAI finish_reason.
_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",  # not expected (tools are disabled) but mapped anyway
    "refusal": "content_filter",
    "model_context_window_exceeded": "length",
    "pause_turn": "stop",
}

_ROLE_LABELS = {
    "user": "Human",
    "assistant": "Assistant",
    "tool": "Tool result",
    "function": "Tool result",
}


def _new_id() -> str:
    return "chatcmpl-" + os.urandom(16).hex()


def _now() -> int:
    return int(time.time())


def _map_finish_reason(stop_reason: str | None, *, default: str = "stop") -> str:
    if not stop_reason:
        return default
    return _FINISH_REASON_MAP.get(stop_reason, default)


def _estimate_tokens(text: str) -> int:
    """Very rough token estimate (~4 chars/token) used only as a fallback when
    the SDK does not report usage. Real counts come from ``ResultMessage.usage``.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _tool_call_id() -> str:
    return "call_" + os.urandom(12).hex()


# =============================================================================
# Tool-call prompt injection + parsing (see README "function calling")
# =============================================================================
# The model is instructed to emit tool calls as a single JSON object. We parse
# that back into OpenAI tool_calls. This is best-effort (text-based), so the
# extractor is deliberately tolerant: it strips markdown fences and scans for
# *every* balanced JSON value, returning the first that is actually tool-call
# shaped. Scanning all candidates (rather than committing to the first balanced
# object or the first fenced block) means an example/non-tool JSON snippet does
# not shadow the real tool call that follows it.

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class ParsedToolCall:
    name: str
    arguments: str  # JSON-encoded string (OpenAI shape)


def _balanced_json(text: str) -> str | None:
    """Return the first balanced ``{...}`` / ``[...]`` substring, honoring string
    literals/escapes so braces inside strings don't throw off the depth count."""
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None


def _iter_balanced_json(text: str) -> "list[str]":
    """Yield each top-level balanced JSON value found in ``text``, in order.

    Advances past each complete value; on an unterminated opener it steps forward
    by one so a complete value nested after the junk can still be recovered.
    """
    spans: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] in "{[":
            span = _balanced_json(text[i:])
            if span:
                spans.append(span)
                i += len(span)
                continue
        i += 1
    return spans


def _iter_json_candidates(text: str) -> "list[str]":
    """Candidate JSON documents from model output: fenced blocks first (the most
    likely place for a clean tool call), then a scan of the whole text."""
    s = text.strip()
    if not s:
        return []
    candidates: list[str] = []
    for match in _FENCE_RE.finditer(s):
        inner = match.group(1).strip()
        if inner:
            candidates.extend(_iter_balanced_json(inner))
    candidates.extend(_iter_balanced_json(s))
    return candidates


def _coerce_arguments(args: object) -> str:
    """Normalize a tool call's arguments to OpenAI's JSON-*string* form.

    OpenAI guarantees ``function.arguments`` is a JSON string that decodes to an
    *object* (clients ``json.loads`` it and splat it as kwargs). So any non-object
    value (scalar, array, or a string that parses to a non-object) is wrapped as
    ``{"value": ...}`` rather than passed through, which would break that contract.
    """
    if isinstance(args, str):
        stripped = args.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return json.dumps({"value": args})
        if isinstance(parsed, dict):
            return stripped
        return json.dumps({"value": parsed})
    if isinstance(args, dict):
        try:
            return json.dumps(args, separators=(",", ":"))
        except (TypeError, ValueError):
            return "{}"
    if args is None:
        return "{}"
    # Scalars / arrays: valid JSON but not an object -> wrap to honor the contract.
    try:
        return json.dumps({"value": args}, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


def _tool_calls_from_data(data: object) -> list[ParsedToolCall] | None:
    """Interpret one parsed JSON value as tool calls, or None if it isn't shaped
    like any of the accepted forms."""
    if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
        raw_calls: object = data["tool_calls"]
    elif isinstance(data, dict) and "name" in data:
        raw_calls = [data]
    elif isinstance(data, list):
        raw_calls = data
    else:
        return None

    calls: list[ParsedToolCall] = []
    for item in raw_calls:  # type: ignore[union-attr]
        if not isinstance(item, dict):
            continue
        # Tolerate {"name","arguments"} and {"function":{"name","arguments"}}.
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = fn.get("arguments", fn.get("parameters", {}))
        calls.append(ParsedToolCall(name=name, arguments=_coerce_arguments(arguments)))
    return calls or None


def _parse_tool_calls(text: str) -> list[ParsedToolCall] | None:
    """Best-effort parse of model output into tool calls; None if it isn't one.

    Accepted shapes (after fence-stripping / balanced extraction):
      * ``{"tool_calls": [{"name": ..., "arguments": {...}}, ...]}``
      * ``{"name": ..., "arguments": {...}}``                (single call)
      * ``[{"name": ..., "arguments": {...}}, ...]``         (bare array)

    Every balanced JSON value in the text is tried in order; the first one that is
    tool-call shaped wins, so a non-tool example earlier in the reply does not
    shadow the real call.
    """
    for candidate in _iter_json_candidates(text):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        calls = _tool_calls_from_data(data)
        if calls:
            return calls
    return None


_TOOL_PROTOCOL = (
    "# Tool use\n"
    "You have access to the tools listed below (as JSON Schemas). When you decide "
    "to call one or more tools, you MUST respond with ONLY a single JSON object and "
    "nothing else — no prose, no explanation, no markdown code fences. Use exactly "
    "this shape:\n\n"
    '{"tool_calls": [{"name": "<tool name>", "arguments": {<arguments as JSON '
    "matching the tool's parameters>}}]}\n\n"
    '- "arguments" must be a JSON object (use {} for a tool that takes no arguments).\n'
    "- You may include more than one entry in the array to call several tools.\n"
    "- If you do NOT need to call a tool, reply to the user normally in plain text "
    "and do NOT output the JSON object.\n"
)


def _build_tool_instructions(
    tools: list[Tool], mode: str, forced_name: str | None
) -> str:
    """Render the tool-use protocol + tool schemas for injection into the system
    prompt."""
    specs = []
    for tool in tools:
        fn = tool.function
        spec: dict = {"name": fn.name}
        if fn.description:
            spec["description"] = fn.description
        spec["parameters"] = (
            fn.parameters
            if fn.parameters is not None
            else {"type": "object", "properties": {}}
        )
        specs.append(spec)

    parts = [_TOOL_PROTOCOL, "Available tools:", json.dumps(specs, indent=2)]
    if mode == "required":
        parts.append(
            "\nFor this request you MUST call at least one of the available tools. "
            "Respond with ONLY the JSON object described above."
        )
    elif mode == "function" and forced_name:
        parts.append(
            f'\nFor this request you MUST call the tool "{forced_name}". '
            "Respond with ONLY the JSON object described above, using that tool."
        )
    return "\n".join(parts)


class ClaudeBackend:
    """Owns the concurrency limit and translates requests to the Agent SDK."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Each query() spawns a CLI subprocess; cap how many run at once.
        self._semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))
        self._timeout = settings.request_timeout

    # ----------------------------------------------------------- translation
    def _split_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[ChatMessage]]:
        """Separate system messages (-> system_prompt) from the conversation."""
        system_parts: list[str] = []
        convo: list[ChatMessage] = []
        for msg in messages:
            if msg.role == "system" or msg.role == "developer":
                text = msg.text
                if text:
                    system_parts.append(text)
            else:
                convo.append(msg)
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, convo

    def _build_prompt(self, convo: list[ChatMessage]) -> str:
        """Fold the non-system conversation into a single prompt string.

        For a single plain user turn we pass the text verbatim (most natural).
        For multi-turn history — including the function-calling round-trip where
        the client echoes back an assistant ``tool_calls`` message followed by
        ``role:"tool"`` results — we render a labeled transcript so prior context
        is preserved across these stateless calls.
        """
        if not convo:
            return ""
        if (
            len(convo) == 1
            and convo[0].role == "user"
            and not convo[0].tool_calls
        ):
            return convo[0].text

        # Resolve tool_call_id -> tool name from any assistant tool_calls so we
        # can label the corresponding tool-result turns meaningfully.
        id_to_name = self._tool_call_names(convo)

        lines: list[str] = []
        for msg in convo:
            label = _ROLE_LABELS.get(msg.role, "Human")
            text = msg.text
            if msg.role in ("tool", "function"):
                name = (
                    id_to_name.get(msg.tool_call_id)
                    or msg.name
                    or "tool"
                )
                lines.append(f"{label} ({name}): {text}")
                continue
            segments: list[str] = []
            if text:
                segments.append(text)
            # Modern `tool_calls`, or the legacy singular `function_call` echo.
            calls = msg.tool_calls
            if not calls and msg.function_call:
                calls = [{"function": msg.function_call}]
            for rendered in self._render_tool_calls(calls):
                segments.append(rendered)
            if not segments:
                continue
            lines.append(f"{label}: " + "\n".join(segments))
        return "\n\n".join(lines)

    @staticmethod
    def _tool_call_names(convo: list[ChatMessage]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for msg in convo:
            for call in msg.tool_calls or []:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                fn = call.get("function")
                name = fn.get("name") if isinstance(fn, dict) else None
                if isinstance(call_id, str) and isinstance(name, str):
                    mapping[call_id] = name
        return mapping

    @staticmethod
    def _render_tool_calls(tool_calls: list[dict] | None) -> list[str]:
        rendered: list[str] = []
        for call in tool_calls or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = fn.get("name") or call.get("name") or "tool"
            args = fn.get("arguments", call.get("arguments", "{}"))
            if args is None:
                # JSON null (some clients serialize a no-arg call this way).
                args = "{}"
            elif not isinstance(args, str):
                with contextlib.suppress(TypeError, ValueError):
                    args = json.dumps(args, separators=(",", ":"))
            rendered.append(f"[Called tool {name} with arguments {args}]")
        return rendered

    def _build_options(
        self,
        *,
        system_prompt: str | None,
        model: str,
        max_tokens: int | None,
        stream: bool,
    ) -> ClaudeAgentOptions:
        """Construct a clean, tool-less, single-turn options object.

        * ``system_prompt`` is set to the client's system content, or left None
          (the SDK then sends an *empty* system prompt — i.e. no agentic
          "Claude Code" persona, just plain text generation).
        * ``allowed_tools=[]`` + ``permission_mode`` that never prompts => no
          file/bash/tool access and no interactive approval.
        * ``max_turns`` is configurable (default > 1): with no tools there is no
          agentic loop, but some models take an internal planning turn before
          answering, and ``max_turns=1`` would abort them with
          "Reached maximum number of turns (1)".
        * ``setting_sources=None`` => do NOT load project/user settings or
          CLAUDE.md, keeping output free of local context.
        """
        env: dict[str, str] = {}
        if max_tokens is not None and max_tokens > 0:
            # The CLI/SDK has no direct max-output-tokens option; this env var is
            # the supported lever. Best-effort — clamped by the model's own cap.
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(max_tokens))

        return ClaudeAgentOptions(
            model=model,
            system_prompt=system_prompt,
            allowed_tools=[],
            disallowed_tools=[],
            max_turns=max(1, self._settings.max_turns),
            permission_mode="bypassPermissions",
            setting_sources=None,
            include_partial_messages=stream,
            env=env,
        )

    def _map_usage(
        self, usage: dict | None, *, prompt_text: str, completion_text: str
    ) -> Usage:
        """Map the SDK usage dict into OpenAI usage, with an estimate fallback."""
        if usage:
            prompt_tokens = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_read_input_tokens", 0) or 0)
                + int(usage.get("cache_creation_input_tokens", 0) or 0)
            )
            completion_tokens = int(usage.get("output_tokens", 0) or 0)
            # If the SDK somehow reported zero output but we have text, estimate.
            if completion_tokens == 0 and completion_text:
                completion_tokens = _estimate_tokens(completion_text)
        else:
            prompt_tokens = _estimate_tokens(prompt_text)
            completion_tokens = _estimate_tokens(completion_text)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    # ------------------------------------------------------------- error map
    @staticmethod
    def _raise_for_assistant_error(error: str | None) -> None:
        if not error:
            return
        mapping = {
            "rate_limit": (429, "rate_limit_error", "rate_limit_exceeded"),
            "billing_error": (402, "billing_error", "billing_error"),
            "invalid_request": (400, "invalid_request_error", None),
            "authentication_failed": (502, "api_error", "backend_auth_failed"),
            "server_error": (502, "api_error", None),
            "unknown": (502, "api_error", None),
        }
        status, etype, code = mapping.get(error, (502, "api_error", None))
        msg = {
            "authentication_failed": (
                "The Claude backend rejected authentication. Re-run the Claude "
                "CLI subscription login (`claude` -> Log in) on the host."
            ),
        }.get(error, f"Claude backend error: {error}.")
        raise GatewayError(msg, status_code=status, type=etype, code=code)

    @staticmethod
    def _raise_for_result_error(result: ResultMessage) -> None:
        if not result.is_error:
            return
        status = result.api_error_status or 502
        detail = ""
        if result.errors:
            detail = "; ".join(str(e) for e in result.errors)
        elif result.result:
            detail = result.result
        message = (
            f"Claude backend reported an error ({result.subtype})"
            + (f": {detail}" if detail else ".")
        )
        etype = "rate_limit_error" if status == 429 else "api_error"
        raise GatewayError(message, status_code=status, type=etype)

    @staticmethod
    def _wrap_sdk_exception(exc: Exception) -> GatewayError:
        """Normalize an SDK exception into a GatewayError.

        Client-facing messages are intentionally generic: the raw exception text
        (which for ``ProcessError`` includes the CLI's full stderr, and may carry
        host paths / diagnostics) is logged server-side only, never sent on the wire.
        """
        # Log the real detail for the operator; clients get a stable message.
        logger.error("Claude backend error: %s: %s", type(exc).__name__, exc)
        if isinstance(exc, CLINotFoundError):
            return GatewayError(
                "Claude Code CLI not found on the gateway host. Install it with "
                "`npm install -g @anthropic-ai/claude-code` and log in once.",
                status_code=500,
                type="api_error",
                code="cli_not_found",
            )
        if isinstance(exc, ProcessError):
            return GatewayError(
                "The Claude CLI process failed.",
                status_code=502,
                type="api_error",
                code="cli_process_error",
            )
        if isinstance(exc, (CLIConnectionError, CLIJSONDecodeError)):
            return GatewayError(
                "Failed to communicate with the Claude CLI.",
                status_code=502,
                type="api_error",
                code="cli_connection_error",
            )
        return GatewayError(
            "Unexpected backend error.",
            status_code=500,
            type="api_error",
        )

    # ---------------------------------------------------------- SDK iteration
    @staticmethod
    async def _safe_aclose(agen) -> None:
        """Tear down the SDK query generator (and its CLI subprocess) without
        blocking the concurrency slot indefinitely.

        ``agen.aclose()`` is shielded so the teardown runs to completion (the SDK
        SIGTERMs/SIGKILLs the subprocess), but we stop *waiting* on it after
        ``_TEARDOWN_TIMEOUT`` so a wedged subprocess can't pin a slot.
        """
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                asyncio.shield(agen.aclose()), timeout=_TEARDOWN_TIMEOUT
            )

    async def _iter_query(self, prompt: str, options: ClaudeAgentOptions):
        """Yield SDK messages from a single ``query()``, applying an *inter-message*
        idle timeout and guaranteeing subprocess teardown.

        The timeout wraps only ``await anext(agen)`` — i.e. the time spent waiting
        on the backend for the next event — and NOT the ``yield``. So a slow client
        reading the stream never counts against the timeout, and a stalled backend
        reliably raises a 504 into the caller's frame (rather than the deadline
        firing while suspended at a yield, where it would be lost).
        """
        agen = query(prompt=prompt, options=options)
        try:
            while True:
                try:
                    async with asyncio.timeout(self._timeout):
                        message = await anext(agen)
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    raise GatewayError(
                        f"Timed out waiting for the model after {self._timeout:g}s.",
                        status_code=504,
                        type="api_error",
                        code="timeout",
                    ) from exc
                except GatewayError:
                    raise
                except (asyncio.CancelledError, GeneratorExit):
                    # Client disconnect / shutdown — propagate so `finally` tears down.
                    raise
                except Exception as exc:  # noqa: BLE001 - normalize SDK errors
                    raise self._wrap_sdk_exception(exc) from exc
                yield message
        finally:
            await self._safe_aclose(agen)

    def _augment_system_for_tools(
        self, base_system: str | None, req: ChatCompletionRequest
    ) -> tuple[str | None, bool]:
        """Append tool-use instructions to the system prompt when tools are
        offered. Returns ``(system_prompt, offering_tools)``.

        ``offering_tools`` is False when no tools are declared or
        ``tool_choice == "none"`` — in that case the system prompt is untouched
        and the gateway behaves exactly as the text-only path.
        """
        tools = req.tool_defs
        mode, forced = req.resolved_tool_choice
        if not tools or mode == "none":
            return base_system, False
        instructions = _build_tool_instructions(tools, mode, forced)
        if base_system:
            return f"{base_system}\n\n{instructions}", True
        return instructions, True

    # ------------------------------------------------------------ non-stream
    async def complete(self, req: ChatCompletionRequest) -> ChatCompletion:
        model_id = self._settings.resolve_model(req.model)
        echo_model = (req.model or self._settings.default_model).strip()
        system_prompt, convo = self._split_messages(req.messages)
        prompt = self._build_prompt(convo)
        system_prompt, offering_tools = self._augment_system_for_tools(
            system_prompt, req
        )
        options = self._build_options(
            system_prompt=system_prompt,
            model=model_id,
            max_tokens=req.effective_max_tokens,
            stream=False,
        )

        text_parts: list[str] = []
        stop_reason: str | None = None
        usage_dict: dict | None = None
        assistant_error: str | None = None

        async with self._semaphore:
            async with contextlib.aclosing(
                self._iter_query(prompt, options)
            ) as messages:
                async for message in messages:
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text_parts.append(block.text)
                        if message.stop_reason:
                            stop_reason = message.stop_reason
                        if message.error:
                            assistant_error = message.error
                    elif isinstance(message, ResultMessage):
                        usage_dict = message.usage
                        if message.stop_reason:
                            stop_reason = message.stop_reason
                        self._raise_for_result_error(message)
                        # Fallback: use the result text if no blocks captured.
                        if not text_parts and message.result:
                            text_parts.append(message.result)

        self._raise_for_assistant_error(assistant_error)

        content = "".join(text_parts)
        usage = self._map_usage(
            usage_dict, prompt_text=prompt, completion_text=content
        )

        tool_calls = _parse_tool_calls(content) if offering_tools else None
        if tool_calls:
            message = ResponseMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=_tool_call_id(),
                        function=FunctionCall(name=c.name, arguments=c.arguments),
                    )
                    for c in tool_calls
                ],
            )
            finish_reason = "tool_calls"
        else:
            message = ResponseMessage(role="assistant", content=content)
            finish_reason = _map_finish_reason(stop_reason)

        return ChatCompletion(
            id=_new_id(),
            created=_now(),
            model=echo_model,
            choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
            usage=usage,
        )

    # ---------------------------------------------------------------- stream
    async def stream_chunks(
        self, req: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Yield OpenAI ``chat.completion.chunk`` objects as text arrives.

        For the text-only path this streams incrementally off the SDK's
        partial-message ``StreamEvent``s — it never buffers the whole reply and
        replays it. When tools are *offered*, the output may be a tool-call JSON
        object, which can't be known until enough has arrived; that path buffers
        the reply, then emits either ``tool_calls`` deltas or the text in one go.
        Raising mid-stream is fine; the SSE layer turns it into an error event.
        """
        model_id = self._settings.resolve_model(req.model)
        echo_model = (req.model or self._settings.default_model).strip()
        system_prompt, convo = self._split_messages(req.messages)
        prompt = self._build_prompt(convo)
        system_prompt, offering_tools = self._augment_system_for_tools(
            system_prompt, req
        )
        options = self._build_options(
            system_prompt=system_prompt,
            model=model_id,
            max_tokens=req.effective_max_tokens,
            stream=True,
        )

        completion_id = _new_id()
        created = _now()

        def _chunk(delta: DeltaMessage, finish_reason: str | None = None,
                   usage: Usage | None = None) -> ChatCompletionChunk:
            return ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=echo_model,
                choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
                usage=usage,
            )

        stop_reason: str | None = None
        usage_dict: dict | None = None
        assistant_error: str | None = None
        streamed_text = False
        streamed_len = 0
        buffer: list[str] = []  # used only when offering_tools

        async with self._semaphore:
            # Initial chunk: announce the assistant role (OpenAI convention). For a
            # tool-call stream OpenAI sends content:null here (not ""), so a strict
            # consumer that branches on `content is None` detects a tool-only turn;
            # the text-only path keeps content="" unchanged.
            yield ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=echo_model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(
                            role="assistant",
                            content=None if offering_tools else "",
                        ),
                        finish_reason=None,
                    )
                ],
            )

            # aclosing() guarantees the SDK generator (and its subprocess) is torn
            # down promptly when the stream ends — including a client disconnect,
            # which closes this generator and unwinds the `async with`.
            async with contextlib.aclosing(
                self._iter_query(prompt, options)
            ) as messages:
                async for message in messages:
                    if isinstance(message, StreamEvent):
                        text = _extract_delta_text(message.event)
                        if text:
                            if offering_tools:
                                buffer.append(text)
                            else:
                                streamed_text = True
                                streamed_len += len(text)
                                yield _chunk(DeltaMessage(content=text))
                    elif isinstance(message, AssistantMessage):
                        if message.stop_reason:
                            stop_reason = message.stop_reason
                        if message.error:
                            assistant_error = message.error
                        # Fallback path: if partial streaming produced nothing
                        # (older CLI), capture the assembled text now.
                        full = "".join(
                            b.text
                            for b in message.content
                            if isinstance(b, TextBlock)
                        )
                        if offering_tools:
                            if not buffer and full:
                                buffer.append(full)
                        elif not streamed_text and full:
                            streamed_len += len(full)
                            yield _chunk(DeltaMessage(content=full))
                    elif isinstance(message, ResultMessage):
                        usage_dict = message.usage
                        if message.stop_reason:
                            stop_reason = message.stop_reason
                        self._raise_for_result_error(message)
                        if offering_tools and not buffer and message.result:
                            buffer.append(message.result)

            # If the backend flagged an assistant-level error, surface it.
            self._raise_for_assistant_error(assistant_error)

            if offering_tools:
                full_text = "".join(buffer)
                streamed_len = len(full_text)
                parsed = _parse_tool_calls(full_text) if full_text else None
                if parsed:
                    for index, call in enumerate(parsed):
                        yield _chunk(
                            DeltaMessage(
                                tool_calls=[
                                    ToolCallDelta(
                                        index=index,
                                        id=_tool_call_id(),
                                        type="function",
                                        function=FunctionCallDelta(
                                            name=call.name, arguments=call.arguments
                                        ),
                                    )
                                ]
                            )
                        )
                    yield _chunk(DeltaMessage(), finish_reason="tool_calls")
                else:
                    if full_text:
                        yield _chunk(DeltaMessage(content=full_text))
                    yield _chunk(
                        DeltaMessage(), finish_reason=_map_finish_reason(stop_reason)
                    )
            else:
                # Final chunk: empty delta + finish_reason.
                finish_reason = _map_finish_reason(stop_reason)
                yield _chunk(DeltaMessage(), finish_reason=finish_reason)

            # Optional usage chunk (OpenAI sends this last when requested).
            if req.wants_usage:
                usage = self._map_usage(
                    usage_dict,
                    prompt_text=prompt,
                    completion_text="x" * streamed_len,  # length-only estimate
                )
                yield ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=echo_model,
                    choices=[],
                    usage=usage,
                )


def _extract_delta_text(event: dict) -> str:
    """Pull incremental text out of a raw Anthropic streaming event.

    The SDK passes the Anthropic Messages-API streaming event through verbatim in
    ``StreamEvent.event``. Text arrives as::

        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "..."}}

    Thinking/tool deltas are ignored (the gateway is text-only).
    """
    if not isinstance(event, dict):
        return ""
    if event.get("type") != "content_block_delta":
        return ""
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return ""
    if delta.get("type") != "text_delta":
        return ""
    text = delta.get("text")
    return text if isinstance(text, str) else ""
