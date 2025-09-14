# syntax=docker/dockerfile:1.4

# --- Stage 1: Builder ---
# This stage has build tools and uses the secret to fetch dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy poetry files and README
COPY pyproject.toml poetry.lock README.md ./

# Copy source code
COPY src/ ./src/

# Install dependencies securely with guaranteed cleanup
RUN --mount=type=secret,id=github_token,env=GITHUB_TOKEN \
    sh -c "trap 'git config --global --remove-section credential 2>/dev/null || true' EXIT; \
    git config --global credential.helper '!f() { test \"$1\" = get && echo \"password=$GITHUB_TOKEN\"; }; f'; \
    poetry config virtualenvs.create false && poetry install --only main"

# --- Stage 2: Final Image ---
# This stage is minimal and contains no build tools or secrets
FROM python:3.12-slim as final

WORKDIR /app

# Create a non-root user for security
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Copy installed packages and application code from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8080

# Run the PostgreSQL adapter
CMD ["python", "-m", "omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.node"]