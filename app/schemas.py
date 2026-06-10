"""Pydantic v2 models for the OpenAI-compatible surface.

Design rules
------------
* Request models use ``extra="ignore"`` so unknown / client-specific fields
  (``user``, ``n``, ``presence_penalty``, ``logit_bias``, ``response_format``,
  ``tools``, ``seed``, ``stream_options`` variants, ...) are accepted and
  silently dropped instead of causing a 422.
* ``content`` on a message may be a plain string OR the OpenAI "parts" array
  (``[{"type": "text", "text": "..."}, ...]``). ``ChatMessage.text`` flattens
  either form to a string (non-text parts such as images are ignored, since the
  gateway is a text-generation profile).
* Response / chunk models mirror OpenAI's wire shapes exactly so the official
  ``openai`` SDK and the wider ecosystem parse them without surprises.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Request models
# =============================================================================
class ChatMessage(BaseModel):
    """A single OpenAI chat message. Tolerant of extra fields."""

    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    # str | parts-array | null (assistant tool-call messages may have null content)
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    # Round-trip fields: assistant turns echo back `tool_calls`; tool-result
    # turns carry `tool_call_id`. Kept as raw dicts (we only read name/args/id).
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    # Legacy singular assistant echo (deprecated OpenAI `function_call`).
    function_call: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        """Flatten ``content`` to plain text."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # OpenAI "parts" array: collect text-bearing parts, ignore images/audio.
        chunks: list[str] = []
        for part in self.content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
            # Some clients send {"type": "input_text", "text": ...} or bare {"text": ...}
            elif part.get("type") == "input_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
            elif "text" in part and isinstance(part["text"], str):
                chunks.append(part["text"])
        return "\n".join(chunks)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include_usage: bool = False


class FunctionDef(BaseModel):
    """A function/tool declaration the client offers the model."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str | None = None
    # JSON Schema describing the arguments; passed through to the prompt verbatim.
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    """OpenAI ``tools[]`` entry. Only ``type:"function"`` is supported.

    ``function`` is optional so entries of other types (e.g. a client sending a
    built-in tool spec) are tolerated and skipped rather than causing a 400.
    """

    model_config = ConfigDict(extra="ignore")

    type: str = "function"
    function: FunctionDef | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions request.

    Only the fields the gateway actually consults are declared; every other
    field the client sends is accepted and ignored (``extra="ignore"``).
    """

    model_config = ConfigDict(extra="ignore")

    # OpenAI rejects an empty messages array with a 400; mirror that here so the
    # request never reaches the CLI with nothing to say.
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None

    # Function calling (modern + legacy spellings). When tools are offered the
    # gateway injects them into the prompt and parses tool calls back out.
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    functions: list[FunctionDef] | None = None  # legacy alias for `tools`
    function_call: str | dict[str, Any] | None = None  # legacy alias for tool_choice

    # Sampling / generation knobs. `max_tokens` (and the newer
    # `max_completion_tokens`) are honored best-effort; the rest are accepted but
    # currently ignored because the Claude CLI exposes no knob for them.
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None

    @property
    def effective_max_tokens(self) -> int | None:
        """`max_completion_tokens` (newer) wins over `max_tokens`."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens

    @property
    def wants_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)

    @property
    def tool_defs(self) -> list[Tool]:
        """Offered tools, from `tools` (preferred) or legacy `functions`."""
        if self.tools:
            return [t for t in self.tools if t.function and t.function.name]
        if self.functions:
            return [Tool(function=fn) for fn in self.functions if fn.name]
        return []

    @property
    def resolved_tool_choice(self) -> tuple[str, str | None]:
        """Normalize tool_choice/function_call to ``(mode, forced_name)``.

        ``mode`` is one of ``"auto"`` (default), ``"none"``, ``"required"``, or
        ``"function"`` (with ``forced_name`` set). Unrecognized values fall back
        to ``"auto"`` so we never error on an unknown choice.
        """
        choice = self.tool_choice if self.tool_choice is not None else self.function_call
        if choice is None:
            return ("auto", None)
        if isinstance(choice, str):
            c = choice.strip().lower()
            if c in ("none", "auto", "required"):
                return (c, None)
            return ("auto", None)
        if isinstance(choice, dict):
            inner = choice.get("function")
            fn = inner if isinstance(inner, dict) else choice
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str) and name:
                return ("function", name)
        return ("auto", None)


# =============================================================================
# Shared response pieces
# =============================================================================
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# =============================================================================
# Tool calls (response side)
# =============================================================================
class FunctionCall(BaseModel):
    name: str
    # OpenAI sends arguments as a JSON-encoded *string*, not an object.
    arguments: str = "{}"


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# =============================================================================
# Non-streaming response
# =============================================================================
class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    # `content` is null when the assistant returns tool calls instead of text.
    content: str | None = ""
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = "stop"
    logprobs: None = None


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None


# =============================================================================
# Streaming response (chunks)
# =============================================================================
class FunctionCallDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    # `index` ties fragments of the same call together across chunks.
    index: int = 0
    id: str | None = None
    type: str | None = None
    function: FunctionCallDelta | None = None


class DeltaMessage(BaseModel):
    # Fields are optional so we can emit role-only, content-only, or empty deltas.
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: str | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    # Present only on the final usage chunk when stream_options.include_usage=true.
    usage: Usage | None = None


# =============================================================================
# Models listing
# =============================================================================
class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "anthropic"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]


# =============================================================================
# Error envelope (OpenAI shape)
# =============================================================================
class ErrorDetail(BaseModel):
    message: str
    type: str = "api_error"
    code: str | None = None
    param: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
