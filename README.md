# Claude Subscription Gateway

[![Build and push Docker image](https://github.com/aikofy/claude-sub-gateway/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/aikofy/claude-sub-gateway/actions/workflows/docker-publish.yml)

A self-hosted **OpenAI-compatible API** that proxies requests to **Claude** through
the [**Claude Agent SDK**](https://github.com/anthropics/claude-agent-sdk-python).
The SDK authenticates using your existing **Claude Code subscription login** —
**no Anthropic Console API key is involved**.

Point any OpenAI-compatible tool (the official `openai` SDKs, LiteLLM, LangChain,
LlamaIndex, OpenWebUI, Continue, Cursor, …) at this gateway with a Bearer key you
define, and the actual model calls flow through your Claude subscription.

> ⚠️ **Personal / internal use only.** This is a thin personal bridge, not a
> reselling or multi-tenant product. Your Claude subscription rate limits apply
> to every call. Not affiliated with Anthropic; your use is subject to
> Anthropic's terms — see the [Disclaimer](#disclaimer).

---

## How the backend auth works (and why there's no API key)

```
  OpenAI client ──HTTP(Bearer)──▶  Gateway (FastAPI)
                                      │  Claude Agent SDK
                                      ▼
                                  Claude Code CLI  ──▶  Claude (your subscription)
                                  (logged in once)
```

* The Agent SDK drives the locally installed **Claude Code CLI**, which you log in
  **once** with your subscription. The SDK reuses that stored auth automatically,
  so **the gateway itself needs no Anthropic API key**.
* The gateway **never** extracts/uses raw OAuth tokens and **never** calls
  `api.anthropic.com` directly — every model call goes through the Agent SDK.

---

## Prerequisites

1. **Python 3.11+**
2. **Node.js 18+** (the Claude Code CLI is a Node package)
3. **Claude Code CLI**, installed and logged in with your subscription:

   ```bash
   npm install -g @anthropic-ai/claude-code

   # Log in once (interactive). Choose "Log in with Claude subscription".
   claude
   # …complete the browser login, then you can quit the CLI.
   ```

   Verify it works without an API key:

   ```bash
   claude -p "say hello"
   ```

   The login is cached (on macOS in the Keychain; on Linux in `~/.claude/`). The
   gateway and the SDK pick it up automatically.

---

## Install & run (local)

```bash
git clone <this repo> claude-subscription-gateway
cd claude-subscription-gateway

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GATEWAY_API_KEYS to one or more secret Bearer keys.

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Generate a strong key:

```bash
python -c "import secrets; print('sk-gw-' + secrets.token_hex(24))"
```

The gateway is now at `http://localhost:8000`, OpenAI base URL
`http://localhost:8000/v1`.

---

## Configuration

All via environment variables (or a `.env` file). See [`.env.example`](.env.example).

| Variable           | Default              | Description |
|--------------------|----------------------|-------------|
| `GATEWAY_API_KEYS` | *(empty)*            | **Required.** Comma-separated Bearer keys clients must send. Empty ⇒ all requests rejected (401). |
| `DEFAULT_MODEL`    | `claude-sonnet-4-6`  | Model used when a request omits `model`. |
| `MODEL_ALIASES`    | *(empty)*            | Extra `name → claude-id` aliases. JSON (`{"gpt-4o":"claude-opus-4-8"}`) or `name:target,…`. Merged over the built-ins. |
| `MAX_CONCURRENCY`  | `4`                  | Max concurrent Claude CLI subprocesses; extra requests queue. |
| `REQUEST_TIMEOUT`  | `600`                | Idle timeout (s): max wait for the next backend event. Trips on a stalled model; a slow client reading a stream does not count against it. |
| `HOST`             | `0.0.0.0`            | Bind address. |
| `PORT`             | `8000`               | Bind port. |
| `LOG_LEVEL`        | `INFO`               | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |
| `CORS_ORIGINS`     | `*`                  | Comma-separated allowed origins, or `*`. |

### Model names

Send a real Claude id, a CLI short name, or a friendly alias — the gateway maps it:

* **Canonical ids** (pass-through): `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`
* **Short names**: `opus`, `sonnet`, `haiku`
* **Built-in friendly aliases**: `gpt-4`, `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini`,
  `gpt-3.5-turbo`, `claude-3-opus`, `claude-3-5-sonnet`, `claude-3-haiku`

Anything unknown is passed straight to the CLI. The **response always echoes the
exact `model` string the client requested.**

---

## Endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /v1/chat/completions` *(and `/chat/completions`)* | Bearer | Chat completions, streaming or not |
| `GET  /v1/models` *(and `/models`)* | Bearer | List available models (OpenAI shape) |
| `GET  /health` | none | Liveness probe |
| `GET  /` | none | Service info |
| `GET  /docs` | none | Interactive OpenAPI docs (FastAPI) |

Errors use OpenAI's envelope: `{"error": {"message", "type", "code", "param"}}`.

---

## Examples

### `curl` — non-streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user",   "content": "Give me three uses for a paperclip."}
    ]
  }'
```

### `curl` — streaming (SSE)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","stream":true,"messages":[{"role":"user","content":"Count to 5."}]}'
```

### Official `openai` Python client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-gw-...your-gateway-key...",
)

# Non-streaming
resp = client.chat.completions.create(
    model="gpt-4o",  # alias → claude-sonnet-4-6; response echoes "gpt-4o"
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": "Write a haiku about caching."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

### Function / tool calling

The gateway implements OpenAI-style function calling. You declare `tools` (and
optionally `tool_choice`); the gateway injects them into the prompt, and Claude's
tool calls come back in the standard OpenAI shape (`message.tool_calls`,
`finish_reason="tool_calls"`, `content: null`). You then run the function and send
the result back as a `role: "tool"` message — exactly the normal OpenAI loop.

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-gw-...")

tools = [{
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
}]

messages = [{"role": "user", "content": "What's the weather in Paris?"}]

# 1) First call — the model asks to call the tool.
resp = client.chat.completions.create(
    model="claude-sonnet-4-6", messages=messages, tools=tools,
)
msg = resp.choices[0].message
assert resp.choices[0].finish_reason == "tool_calls"   # msg.content is None
call = msg.tool_calls[0]
args = json.loads(call.function.arguments)             # {"location": "Paris"}

# 2) You execute the function, then send the result back.
messages.append(msg)                                   # the assistant tool-call turn
messages.append({
    "role": "tool",
    "tool_call_id": call.id,
    "content": json.dumps({"temp_c": 21, "sky": "clear"}),
})
final = client.chat.completions.create(
    model="claude-sonnet-4-6", messages=messages, tools=tools,
)
print(final.choices[0].message.content)                # "It's 21°C and clear in Paris."
```

`tool_choice` is honored: `"auto"` (default — call a tool if useful), `"none"`
(never call; behaves like a plain text request), `"required"` (must call some
tool), or `{"type": "function", "function": {"name": "get_weather"}}` (must call
that specific tool). The legacy `functions` / `function_call` fields are accepted
as aliases too.

A minimal `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Weather in Paris?"}],
    "tool_choice": "auto",
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }]
  }'
```

> **Note:** tool calling is implemented by prompt injection (the Claude Code CLI
> exposes no native client-function API), so it is **best-effort** — see the
> [Limitations](#limitations). When tools are offered, a *streaming* request
> (`stream: true`) buffers the reply and then emits the `tool_calls` (or the text)
> in the stream, because the gateway can't know mid-stream whether the output will
> be a tool call. Plain requests without `tools` still stream token-by-token.

### Other ecosystem clients

All of these "just work" by pointing `base_url` at `http://localhost:8000/v1`:

* **LiteLLM** — provider `openai`, set `api_base`:
  ```python
  import litellm
  litellm.completion(
      model="openai/claude-sonnet-4-6",
      api_base="http://localhost:8000/v1",
      api_key="sk-gw-...",
      messages=[{"role": "user", "content": "hi"}],
  )
  ```
* **LangChain** — `ChatOpenAI(base_url="http://localhost:8000/v1", api_key="sk-gw-...", model="claude-sonnet-4-6")`
* **LlamaIndex** — `OpenAI(api_base="http://localhost:8000/v1", api_key="sk-gw-...", model="claude-sonnet-4-6")`
* **OpenWebUI / Continue.dev / Cursor** — add an "OpenAI-compatible" connection
  with the base URL and key. `GET /v1/models` populates the model picker.

---

## Docker

### Pull the prebuilt image (GHCR)

CI publishes a multi-arch image (`linux/amd64`, `linux/arm64`) to the GitHub
Container Registry on every push to `main` and on version tags:

```bash
docker pull ghcr.io/aikofy/claude-sub-gateway:latest
```

Use that image name in the `docker run` commands below in place of
`claude-gateway`. If the package is private, authenticate first with a GitHub
token that has the `read:packages` scope:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <your-github-username> --password-stdin
```

### Build it yourself

```bash
docker build -t claude-gateway .
```

The image contains Node + the Claude CLI but **not** your login. Provide it at
runtime via a volume mounted at `/home/appuser/.claude`.

**Option A — log in inside the container (works from any host, incl. macOS):**

```bash
docker volume create claude-login

# One-time interactive login into the persistent volume:
docker run --rm -it -v claude-login:/home/appuser/.claude claude-gateway claude
#   → choose "Log in with Claude subscription", finish the browser flow, quit.

# Run the gateway, reusing that login:
docker run --rm -p 8000:8000 \
  -e GATEWAY_API_KEYS="sk-gw-yourkey" \
  -v claude-login:/home/appuser/.claude \
  claude-gateway
```

**Option B — reuse a Linux host login:** mount your host's `~/.claude`
(contains `.credentials.json`) at `/home/appuser/.claude`. (macOS stores the
login in the Keychain, which can't be mounted — use Option A there.)

> For fully headless hosts you can also create a long-lived token with
> `claude setup-token` (stored under `~/.claude`); the same volume-mount approach
> then applies.

---

## Testing

The whole suite **mocks the Agent SDK**, so no live subscription or CLI subprocess
is needed.

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage includes: auth pass/fail (OpenAI 401 envelope), non-streaming and real
incremental streaming completions, `/v1/models`, unknown-field tolerance, model
aliasing, usage/finish-reason mapping, **function/tool calling** (parsing, the
result round-trip, streaming and non-streaming tool-call shapes, `tool_choice`),
and an end-to-end check driving a live server with the **official `openai`
client** (non-stream, stream, models, 401).

---

## Limitations

* **Sampling knobs are partial.** `max_tokens` / `max_completion_tokens` are honored
  best-effort via the CLI's `CLAUDE_CODE_MAX_OUTPUT_TOKENS`. `temperature`, `top_p`,
  `stop`, `presence_penalty`, etc. are **accepted but ignored** — the Claude Code CLI
  exposes no knob for them. (Nothing errors; they're silently dropped.)
* **Token usage** comes from the SDK's `ResultMessage.usage` when available; if it's
  missing, `usage` is a rough character-based estimate (~4 chars/token).
* **Stateless / single-turn.** Each request is an independent `query()` with
  `max_turns=1` and no tools. Multi-turn history you send in `messages` is folded
  into one prompt with role markers — there is no server-side session store.
* **Function calling is best-effort.** It works (see [Function / tool
  calling](#function--tool-calling)), but it's implemented by injecting the tool
  schemas into the prompt and parsing the model's reply — not via a native
  function API (the gateway still runs `allowed_tools=[]`; the CLI's own agentic
  tools are never executed). So: the model may occasionally answer in prose when a
  tool call was expected (or vice-versa), and very large/complex JSON-Schema
  `parameters` are conveyed only as well as the model follows them. `strict` mode
  is not enforced. With tools offered, streaming responses are buffered (the
  tool-call vs. text decision can't be known mid-stream); plain text requests
  without `tools` still stream incrementally.
* **Not multimodal.** Image/audio content parts are dropped; only text is sent.
* **Rate limits are your subscription's.** Concurrency is capped by
  `MAX_CONCURRENCY` (each call spawns a CLI subprocess); excess requests queue.
  A client disconnect cancels the in-flight query (both streaming and
  non-streaming) and frees the slot.
* **One choice per request** (`n` is ignored; `choices` always has length 1).

---

## Project layout

```
app/
  main.py            FastAPI app, routes, CORS, exception handlers, SSE
  auth.py            Bearer-key verification (constant-time)
  schemas.py         Pydantic v2 OpenAI request/response/chunk/model models
  claude_backend.py  Agent SDK calls + OpenAI⇄SDK translation
  config.py          pydantic-settings configuration
  errors.py          GatewayError + OpenAI error envelope helpers
tests/               pytest suite (Agent SDK mocked)
Dockerfile           Python + Node + Claude CLI
.env.example         Configuration template
```

## License

[MIT](LICENSE) © 2026 aikofy. Use with your own Claude subscription, within
Anthropic's terms — see the [Disclaimer](#disclaimer) below.

## Disclaimer

This project is **not affiliated with, endorsed by, or sponsored by Anthropic.**
"Claude" and "Claude Code" are trademarks of Anthropic, PBC, used here only
nominatively to describe compatibility.

Using your Claude subscription through this gateway is **subject to Anthropic's
terms** — including the Consumer Terms of Service, the Usage Policy, and any
Claude Code / subscription terms. You are solely responsible for ensuring your
use complies with them. This gateway is intended for **personal / internal use
only**; it is not a reselling or multi-tenant product, and routing a consumer
subscription through a programmatic API may not be permitted for all plans —
check before you rely on it.

The software is provided **"as is", without warranty of any kind** (see
[`LICENSE`](LICENSE)). The authors are not liable for any account action,
service interruption, rate-limiting, or other consequence arising from its use.
