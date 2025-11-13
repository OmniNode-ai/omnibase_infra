# Multi-stage Dockerfile for metadata stamping service
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

# System dependencies
# BuildKit cache mounts persist APT cache between builds for faster subsequent builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    # Required for building Python packages
    build-essential \
    gcc \
    # Required for PostgreSQL client
    libpq-dev \
    # Utilities
    curl \
    git

# Install Poetry
ENV POETRY_VERSION=2.1.3
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Development stage
FROM base as development

# Set PYTHONPATH to include source directory
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Copy Poetry configuration
COPY pyproject.toml poetry.lock README.md ./

# Install development dependencies with GitHub authentication for private repos
# BuildKit cache mount persists Poetry cache between builds (wheels, metadata) for faster installs
RUN --mount=type=secret,id=gh_pat \
    --mount=type=cache,target=/tmp/poetry_cache \
    export GIT_TOKEN=$(cat /run/secrets/gh_pat 2>/dev/null || echo "") && \
    if [ -n "$GIT_TOKEN" ]; then \
        # Prevent token exposure in logs
        export GIT_ASKPASS=/bin/true && \
        export POETRY_NO_INTERACTION=1 && \
        git config --global url."https://${GIT_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi && \
    poetry install --with dev && \
    if [ -n "$GIT_TOKEN" ]; then \
        git config --global --unset url."https://github.com/".insteadOf || true; \
    fi

# Copy source code
COPY . .

# Create non-root user for development
RUN useradd --create-home --shell /bin/bash dev && \
    chown -R dev:dev /app
USER dev

EXPOSE 8053 9090

# Development command with hot reload
# Note: PYTHONPATH includes /app/src, so import path is omninode_bridge.* not src.omninode_bridge.*
CMD ["poetry", "run", "uvicorn", "omninode_bridge.services.metadata_stamping.main:app", \
     "--host", "0.0.0.0", "--port", "8053", "--reload", \
     "--log-level", "debug"]

# Production stage
FROM base as production

# Set PYTHONPATH to include source directory
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Copy Poetry configuration
COPY pyproject.toml poetry.lock README.md ./

# Copy source code BEFORE poetry install (needed for current project installation)
COPY src/ ./src/
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Install only production dependencies with GitHub authentication
# BuildKit cache mount persists Poetry cache between builds (wheels, metadata) for faster installs
RUN --mount=type=secret,id=gh_pat \
    --mount=type=cache,target=/tmp/poetry_cache \
    export GIT_TOKEN=$(cat /run/secrets/gh_pat 2>/dev/null || echo "") && \
    if [ -n "$GIT_TOKEN" ]; then \
        # Prevent token exposure in logs
        export GIT_ASKPASS=/bin/true && \
        export POETRY_NO_INTERACTION=1 && \
        git config --global url."https://${GIT_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi && \
    poetry install --only=main && \
    if [ -n "$GIT_TOKEN" ]; then \
        git config --global --unset url."https://github.com/".insteadOf || true; \
    fi

# Create non-root user for production
RUN useradd --create-home --shell /bin/bash stamping && \
    chown -R stamping:stamping /app
USER stamping

EXPOSE 8053 9090

# Health check - uses simple /health endpoint for fast response
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8053/health || exit 1

# Production command
# Note: PYTHONPATH includes /app/src, so import path is omninode_bridge.* not src.omninode_bridge.*
CMD ["poetry", "run", "uvicorn", "omninode_bridge.services.metadata_stamping.main:app", \
     "--host", "0.0.0.0", "--port", "8053", \
     "--workers", "4", "--log-level", "info"]
