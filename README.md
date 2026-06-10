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
| `MAX_TURNS`        | `8`                  | Max agent turns per request. Keep **> 1** — some models (e.g. Haiku) take an internal planning turn, and `1` makes the SDK abort with "Reached maximum number of turns (1)" (a 502). No tools are enabled, so this never causes a tool loop. |
| `REQUEST_TIMEOUT`  | `600`                | Idle timeout (s): max wait for the next backend event. Trips on a stalled model; a slow client reading a stream does not count against it. |
| `HOST`             | `0.0.0.0`            | Bind address. |
| `PORT`             | `8000`               | Bind port. |
| `LOG_LEVEL`        | `INFO`               | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |
| `CORS_ORIGINS`     | `*`                  | Comma-separated allowed origins, or `*`. |

### Model names

Send a real Claude id, a CLI short name, or a friendly alias — the gateway maps it:

* **Canonical ids** (pass-through): `claude-fable-5`, `claude-opus-4-8`,
  `claude-sonnet-4-6`, `claude-haiku-4-5`
* **Short names**: `fable`, `opus`, `sonnet`, `haiku`
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

### Docker Compose (recommended)

Compose wires up the persistent login volume for you, so you **log in once and
never again** — the `claude-login` volume survives `down`, restarts, and image
rebuilds (only `docker compose down -v` removes it).

```bash
cp .env.example .env                      # set GATEWAY_API_KEYS
docker compose run --rm gateway claude    # one-time interactive subscription login
#   → choose "Log in with Claude subscription", finish the browser flow, quit.
docker compose up -d                      # gateway at http://localhost:8000
```

The login is written to the named volume during that `run` step, so every
`compose up` afterwards reuses it. Edit [`docker-compose.yml`](docker-compose.yml)
to use the prebuilt GHCR image instead of building locally.

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

> The image declares a **default volume** at `/home/appuser/.claude`, so if you
> forget the `-v` flag your login still isn't written into the container's
> writable layer. But that fallback is an *anonymous* volume — it's per-container
> and is **deleted by `docker run --rm`**. To actually "log in once and reuse it",
> mount a **named** volume (below) or use Docker Compose (above). Don't rely on
> the default for persistence.

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

## Kubernetes

Deploy either with the **Helm chart** (recommended) or the **plain manifests**.
Both encode the same essentials: a single replica, a PVC for the login, the
non-root `fsGroup`, and the streaming-friendly Ingress annotations.

### Helm chart (recommended)

The chart is in [`deploy/helm/claude-sub-gateway`](deploy/helm/claude-sub-gateway)
and is published to GHCR as an OCI artifact on each `vX.Y.Z` tag.

```bash
# From the registry (released versions):
helm install claude-gateway \
  oci://ghcr.io/aikofy/charts/claude-sub-gateway --version <X.Y.Z> \
  --set gatewayApiKeys="sk-gw-yourkey"

# …or straight from a checkout of this repo:
helm install claude-gateway deploy/helm/claude-sub-gateway \
  --set gatewayApiKeys="sk-gw-yourkey"

# then log in once (the chart's NOTES print this too):
kubectl exec -it deploy/claude-gateway -- claude
```

Common values (`--set` or a `-f values.yaml`):

| Value | Default | Purpose |
|---|---|---|
| `gatewayApiKeys` | *(required)* | Comma-separated Bearer keys (a Secret is created) |
| `existingSecret` | `""` | Use a Secret you manage instead of `gatewayApiKeys` |
| `image.tag` | chart `appVersion` | e.g. `latest` to track `main` |
| `config.defaultModel` / `config.maxConcurrency` / `config.maxTurns` / `config.requestTimeout` | sonnet / 4 / 8 / 600 | Gateway env (`maxTurns` must stay > 1) |
| `persistence.size` / `persistence.storageClass` / `persistence.existingClaim` | `1Gi` / default / `""` | Login PVC |
| `ingress.enabled` / `ingress.hosts` / `ingress.tls` | `false` | TLS Ingress (SSE annotations preset) |
| `resources` | 250m/512Mi → 2/2Gi | CPU / memory |

To expose it over HTTPS, set `ingress.enabled=true` with your host/TLS (the
chart ships the streaming annotations; add a cert-manager `ClusterIssuer`
annotation via `ingress.annotations` if you use one):

```bash
helm upgrade --install claude-gateway deploy/helm/claude-sub-gateway \
  --set gatewayApiKeys="sk-gw-yourkey" \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=gateway.example.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=Prefix \
  --set ingress.tls[0].secretName=claude-gateway-tls \
  --set ingress.tls[0].hosts[0]=gateway.example.com
```

### Plain manifests

Ready-to-apply manifests are in
[`deploy/k8s/claude-gateway.yaml`](deploy/k8s/claude-gateway.yaml): a `Secret`
(your Bearer keys), a `PersistentVolumeClaim` for the Claude login, a
single-replica `Deployment`, and a `ClusterIP` `Service`.

```bash
# 1. Edit GATEWAY_API_KEYS in the Secret first, then apply:
kubectl apply -f deploy/k8s/claude-gateway.yaml

# 2. Log in to your Claude subscription ONCE. Creds are written to the PVC, so
#    they survive pod restarts and rollouts — you won't repeat this.
kubectl exec -it deploy/claude-gateway -- claude
#    → choose "Log in with Claude subscription", open the printed URL in your
#      browser, finish the flow, then quit the CLI.

# 3. (optional) confirm the subscription works from inside the pod:
kubectl exec -it deploy/claude-gateway -- claude -p "say hi"

# 4. Reach it (port-forward for a quick test):
kubectl port-forward svc/claude-gateway 8000:80
curl -H "Authorization: Bearer <your-key>" http://localhost:8000/v1/models
```

No pod restart is needed after logging in — each request spawns a fresh CLI
process that picks up the stored credentials.

### Ingress (HTTPS)

[`deploy/k8s/ingress.yaml`](deploy/k8s/ingress.yaml) exposes the gateway over TLS.
It assumes the **NGINX ingress controller** and **cert-manager** (automatic
Let's Encrypt). Edit the host and `ClusterIssuer`, then apply:

```bash
kubectl apply -f deploy/k8s/ingress.yaml
# clients then point at https://gateway.example.com/v1 as the OpenAI base URL:
curl -H "Authorization: Bearer <your-key>" https://gateway.example.com/v1/models
```

**Bring your own certificate** instead of cert-manager? Remove the
`cert-manager.io/cluster-issuer` annotation and create the TLS secret yourself:

```bash
kubectl create secret tls claude-gateway-tls --cert=tls.crt --key=tls.key
```

The example sets **streaming-critical** annotations for this gateway:
`proxy-buffering: "off"` so SSE chunks flush immediately instead of being held to
the end; generous `proxy-read-timeout` / `proxy-send-timeout` (keep them ≥
`REQUEST_TIMEOUT`, since nginx defaults to 60s and would otherwise truncate long
completions); and `proxy-body-size: "10m"` for large prompts. On a non-NGINX
controller (Traefik, HAProxy, a cloud LB) translate these to its equivalents —
otherwise streaming can appear to "hang" until the response completes.

### Deploying on k3s / Traefik

k3s (and some other distros) ship **Traefik** as the default ingress controller,
not NGINX — but the chart and plain manifests default to `ingressClassName: nginx`.
If the class doesn't match your controller the Ingress is silently **ignored**:
HTTPS may still terminate (via a wildcard/existing cert) yet every request returns
a plain-text **`404 page not found`** (Traefik's default backend) instead of the
gateway's JSON. Set the class to match:

```bash
kubectl get ingressclass            # find the real NAME, e.g. "traefik"

helm upgrade --install claude-gateway \
  oci://ghcr.io/aikofy/charts/claude-sub-gateway --version <X.Y.Z> -n <namespace> \
  --set gatewayApiKeys="sk-gw-yourkey" \
  --set ingress.enabled=true \
  --set ingress.className=traefik \
  --set ingress.hosts[0].host=gateway.example.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=Prefix \
  --set ingress.tls[0].secretName=gateway-tls \
  --set ingress.tls[0].hosts[0]=gateway.example.com
```

- Traefik **ignores** the chart's `nginx.ingress.kubernetes.io/*` annotations —
  harmless, since Traefik doesn't buffer responses, so SSE streaming works anyway.
- Confirm the gateway is healthy independent of ingress:
  `kubectl port-forward -n <namespace> svc/claude-gateway 8000:80`, then
  `curl -H "Authorization: Bearer <key>" http://localhost:8000/v1/models`.
- TLS: `ingress.tls[].secretName` must be populated by cert-manager (add a
  `cert-manager.io/cluster-issuer` via `ingress.annotations`) or be a secret you
  pre-create (e.g. a wildcard cert) — otherwise HTTPS has no certificate.

> **Pulling from GHCR and getting a 401?** The packages are still private — a
> *public repo does not make its packages public*. Make both
> `charts/claude-sub-gateway` (chart) and `claude-sub-gateway` (image) public in
> your org's **Packages** settings, or `helm registry login ghcr.io` and add an
> `imagePullSecret`. See [Helm chart](#helm-chart-recommended).

### Notes specific to Kubernetes

- **Run a single replica.** The PVC is `ReadWriteOnce` (one pod at a time), and
  there is one subscription login behind the whole service — multiple replicas
  would multiply rate-limit pressure against one account. The Deployment pins
  `replicas: 1` with `strategy: Recreate` (a RWO volume can't attach to the old
  and new pods during a rolling update). Scale throughput with `MAX_CONCURRENCY`,
  not replicas.
- **The PVC *is* the login persistence.** Creds live at `/home/appuser/.claude`
  on the `claude-login` PVC. Deleting that PVC (e.g. `kubectl delete -f …`) loses
  the login and you must redo step 2.
- **Ownership.** The pod sets `fsGroup: 10001` / `runAsUser: 10001` so the mounted
  volume is writable by the image's non-root `appuser`; don't change these unless
  you rebuild the image with a different uid.
- **Ready ≠ logged in.** `/health` (the probes) doesn't call Claude, so the pod
  goes Ready before login — but completions fail until step 2 is done.
- **Private image?** If the GHCR package is private, create a pull secret and
  uncomment `imagePullSecrets` in the manifest:
  ```bash
  kubectl create secret docker-registry ghcr-pull \
    --docker-server=ghcr.io \
    --docker-username=<github-username> \
    --docker-password=<token-with-read:packages>
  ```
- Pods are Linux, so the in-pod `claude` login writes a normal
  `~/.claude/.credentials.json` (no macOS Keychain involved).

> Same caveat as everywhere: this routes a **personal** Claude subscription —
> keep it to internal/single-tenant use and within Anthropic's terms (see the
> [Disclaimer](#disclaimer)).

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
* **Stateless.** Each request is an independent `query()` with no tools and a
  bounded number of internal turns (`MAX_TURNS`, default 8 — with no tools there
  is no agentic loop). Multi-turn history you send in `messages` is folded into
  one prompt with role markers — there is no server-side session store.
* **A non-system message is required.** `messages` must contain at least one
  non-system message with non-empty content; otherwise the gateway returns a
  400 (the underlying CLI cannot run an empty prompt).
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
deploy/
  k8s/               plain Kubernetes manifests (+ optional TLS Ingress)
  helm/              Helm chart (published to GHCR as an OCI artifact)
.github/workflows/   CI: Docker image + Helm chart publishing to GHCR
Dockerfile           Python + Node + Claude CLI
docker-compose.yml   Compose file with a persistent login volume
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
