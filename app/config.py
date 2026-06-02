"""Configuration for the Claude Subscription Gateway.

Everything is driven by environment variables (optionally loaded from a `.env`
file) via ``pydantic-settings``.

Notes on parsing
----------------
``pydantic-settings`` tries to JSON-decode any field typed as a ``list``/``dict``
*before* validators run, which makes plain ``KEY=a,b,c`` env values raise errors.
To keep the env format friendly (comma-separated lists, ``k:v`` or JSON maps) we
store the raw values as strings and expose parsed views through cached
properties. This is robust across pydantic-settings versions.
"""

from __future__ import annotations

import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Canonical Claude model ids -------------------------------------------------
# These are passed straight through to the Claude CLI's ``--model`` flag. The CLI
# also accepts short aliases like "opus"/"sonnet"/"haiku", which we expose below.
CANONICAL_MODELS: tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)

# --- Built-in friendly aliases --------------------------------------------------
# Maps an OpenAI-ish (or short) model name -> a real Claude model id. Users can
# extend / override these via the MODEL_ALIASES env var. Anything not found here
# is passed through to the CLI verbatim (so real Claude ids and CLI short names
# like "opus" still work).
DEFAULT_ALIASES: dict[str, str] = {
    # Short Claude names understood by the CLI, kept explicit for /v1/models.
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    # OpenAI-style names so drop-in clients that hardcode them still work.
    "gpt-4": "claude-opus-4-8",
    "gpt-4o": "claude-sonnet-4-6",
    "gpt-4-turbo": "claude-sonnet-4-6",
    "gpt-4o-mini": "claude-haiku-4-5",
    "gpt-3.5-turbo": "claude-haiku-4-5",
    # Legacy Claude names some tools still send.
    "claude-3-opus": "claude-opus-4-8",
    "claude-3-5-sonnet": "claude-sonnet-4-6",
    "claude-3-haiku": "claude-haiku-4-5",
}


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Comma-separated list of accepted client API keys (Bearer tokens).
    gateway_api_keys: str = ""

    # Model used when the client omits `model` or sends an empty value.
    default_model: str = "claude-sonnet-4-6"

    # Extra alias mappings. Either JSON (`{"my-model":"claude-opus-4-8"}`) or a
    # comma-separated `name:target,name2:target2` string. Merged over DEFAULT_ALIASES.
    model_aliases: str = ""

    # Max number of concurrent Claude CLI subprocesses. Requests beyond this queue.
    max_concurrency: int = 4

    # Maximum agent turns per request. MUST be > 1: some models (notably Haiku)
    # take an internal planning/thinking turn before answering, and max_turns=1
    # makes the SDK abort with "Reached maximum number of turns (1)" (surfaced as
    # a 502). Because allowed_tools=[] there is no tool loop, so a higher cap only
    # lets the model finish its single user-facing reply — it never calls tools or
    # runs away.
    max_turns: int = 8

    # Idle timeout in seconds: the maximum time to wait for the *next* event from
    # the backend. A stalled model trips it; a slow client reading the stream does
    # not (it only bounds backend production, not client consumption). For
    # non-streaming requests the whole reply arrives as one event, so this is
    # effectively the total generation budget.
    request_timeout: float = 600.0

    # Server bind address.
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging level for the gateway (DEBUG/INFO/WARNING/ERROR).
    log_level: str = "INFO"

    # Comma-separated list of allowed CORS origins. "*" allows any origin.
    cors_origins: str = "*"

    # ------------------------------------------------------------------ helpers
    @property
    def api_keys(self) -> list[str]:
        """Accepted Bearer keys, de-duplicated, in order."""
        keys: list[str] = []
        for raw in self.gateway_api_keys.split(","):
            key = raw.strip()
            if key and key not in keys:
                keys.append(key)
        return keys

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins or ["*"]

    @property
    def alias_table(self) -> dict[str, str]:
        """DEFAULT_ALIASES merged with any user-provided MODEL_ALIASES."""
        table = dict(DEFAULT_ALIASES)
        table.update(self._parse_user_aliases())
        return table

    def _parse_user_aliases(self) -> dict[str, str]:
        raw = self.model_aliases.strip()
        if not raw:
            return {}
        # Prefer JSON if it looks like a JSON object.
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
                return {str(k).strip(): str(v).strip() for k, v in data.items()}
            except (json.JSONDecodeError, AttributeError):
                return {}
        # Fall back to `name:target,name2:target2`.
        parsed: dict[str, str] = {}
        for pair in raw.split(","):
            if ":" not in pair:
                continue
            name, _, target = pair.partition(":")
            name, target = name.strip(), target.strip()
            if name and target:
                parsed[name] = target
        return parsed

    def resolve_model(self, requested: str | None) -> str:
        """Map a client-supplied model name to a real Claude model id.

        Resolution order: empty -> DEFAULT_MODEL; known alias -> mapped id;
        otherwise pass the value straight through to the CLI.
        """
        if not requested or not requested.strip():
            return self.default_model
        requested = requested.strip()
        return self.alias_table.get(requested, requested)

    def advertised_models(self) -> list[str]:
        """Model ids surfaced by GET /v1/models (canonical ids + alias names)."""
        seen: list[str] = []
        for name in (*CANONICAL_MODELS, *self.alias_table.keys()):
            if name not in seen:
                seen.append(name)
        return seen


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (cached)."""
    return Settings()
