# Multi-stage Dockerfile for Deployment Receiver Node
# Optimized for production with security hardening and Docker-in-Docker support

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.12-slim AS builder

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true

# Install system dependencies and Poetry
# BuildKit cache mounts persist APT cache between builds for faster subsequent builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && pip install poetry==${POETRY_VERSION} \
    && apt-get clean

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only dependency files first (for better layer caching)
COPY pyproject.toml ./

# Copy poetry.lock if it exists (using wildcard pattern to avoid error if missing)
COPY poetry.loc[k] ./poetry.lock* ./

# Install dependencies from GitHub repositories (requires authentication via BuildKit secret)
# BuildKit cache mount persists Poetry cache between builds (wheels, metadata) for faster installs
# BuildKit secret mount provides GITHUB_TOKEN for git authentication
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=secret,id=github_token \
    if [ -f /run/secrets/github_token ]; then \
        export GITHUB_TOKEN=$(cat /run/secrets/github_token); \
        git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi && \
    if [ ! -f poetry.lock ]; then \
        echo "poetry.lock not found, generating from pyproject.toml..."; \
        poetry lock; \
    fi && \
    poetry install --only=main --no-root

# ============================================================================
# Stage 2: Runtime - Minimal production image with Docker support
# ============================================================================
FROM python:3.12-slim AS runtime

# Metadata labels
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="OmniNode Team" \
      org.opencontainers.image.url="https://github.com/omninode-ai/omninode_bridge" \
      org.opencontainers.image.source="https://github.com/omninode-ai/omninode_bridge" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="OmniNode Bridge - Deployment Receiver Node" \
      org.opencontainers.image.description="ONEX Deployment Receiver Node for remote container deployment" \
      org.opencontainers.image.licenses="MIT"

# Runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies only (including Docker CLI for Docker operations)
# BuildKit cache mounts persist APT cache between builds for faster subsequent builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    docker-ce-cli \
    && apt-get clean

# Create non-root user for security
# Note: User must be in docker group to access Docker socket (configured at runtime)
RUN groupadd -r deployment && \
    useradd -r -g deployment -u 1000 deployment

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --chown=deployment:deployment src/ /app/src/

# Create directories for logs, data, and deployment packages
RUN mkdir -p /app/logs /app/data /tmp/deployment_packages && \
    chown -R deployment:deployment /app /tmp/deployment_packages

# Switch to non-root user
USER deployment

# Health check using FastAPI health endpoint
# Verifies node is running and Docker daemon is accessible
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports for REST API
EXPOSE 8001 9091

# Default command - run FastAPI server with uvicorn (standalone mode)
CMD ["python", "-m", "uvicorn", "omninode_bridge.nodes.deployment_receiver_effect.v1_0_0.main_standalone:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
