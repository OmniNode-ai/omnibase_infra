FROM python:3.12-slim

LABEL maintainer="OmniNode Team <team@omninode.ai>"
LABEL description="OnexTree Service - Fast project structure intelligence"

# Install system dependencies
# BuildKit cache mounts persist APT cache between builds for faster subsequent builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    git \
    curl

# Install Poetry
RUN pip install --no-cache-dir poetry==2.1.3

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock README.md ./

# Copy source code BEFORE poetry install (needed for current project installation)
COPY src/ ./src/

# Install dependencies with GitHub authentication for private repos
# BuildKit cache mount persists Poetry cache between builds (wheels, metadata) for faster installs
RUN --mount=type=secret,id=gh_pat \
    --mount=type=cache,target=/root/.cache/pypoetry \
    export GIT_TOKEN=$(cat /run/secrets/gh_pat 2>/dev/null || echo "") && \
    if [ -n "$GIT_TOKEN" ]; then \
        # Prevent token exposure in logs
        export GIT_ASKPASS=/bin/true && \
        export POETRY_NO_INTERACTION=1 && \
        git config --global url."https://${GIT_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-interaction --no-ansi && \
    if [ -n "$GIT_TOKEN" ]; then \
        git config --global --unset url."https://github.com/".insteadOf || true; \
    fi

# Create non-root user with home directory and set ownership
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Change to src directory so onextree_service can be found
WORKDIR /app/src

# Expose service port
EXPOSE 8058

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8058/health || exit 1

# Run service
CMD ["uvicorn", "onextree_service.main:app", "--host", "0.0.0.0", "--port", "8058"]
