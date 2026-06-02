# Claude Subscription Gateway
#
# Bundles Python + Node.js + the Claude Code CLI. The gateway needs NO Anthropic
# API key — it relies on the CLI's stored *subscription* login. That login is NOT
# baked into the image; you must supply it at runtime (see README "Docker"):
#   * mount a persistent volume at /home/appuser/.claude and run `claude` login
#     once inside the container, OR
#   * mount a host directory that already holds a Linux Claude login
#     (~/.claude with .credentials.json).
# macOS Keychain credentials do NOT transfer into the container.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_MAJOR=20

# --- System deps: Node.js (for the Claude Code CLI) + the CLI itself ----------
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force \
    && apt-get purge -y gnupg \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (cached layer) ----------------------------------------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- App ----------------------------------------------------------------------
COPY app ./app

# Run as non-root. (Claude's "bypassPermissions" mode refuses to run as root;
# a normal user avoids that and is good practice anyway.) The CLI stores its
# subscription login under $HOME/.claude. Pre-create that dir owned by appuser
# so any volume mounted there inherits the right ownership (an empty named volume
# is initialized from the image dir, which would otherwise be root-owned).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.claude \
    && chown -R appuser:appuser /app /home/appuser/.claude

# Default (anonymous) volume for the login so a `docker run` that FORGETS the
# mount still keeps credentials out of the container's writable layer. This is
# only a fallback: an anonymous volume is per-container and is deleted by
# `docker run --rm`. For "log in once, reuse forever" mount a NAMED volume here
# (or use docker compose — see README); a named mount takes precedence over this.
VOLUME ["/home/appuser/.claude"]

USER appuser
ENV HOME=/home/appuser

EXPOSE 8000
ENV HOST=0.0.0.0 \
    PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# Shell form so $HOST/$PORT overrides are honored at runtime.
CMD ["sh", "-c", "exec uvicorn app.main:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\""]
