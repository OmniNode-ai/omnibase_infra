# Multi-stage build for O.N.E. v0.1 Metadata Stamping Service
FROM python:3.12-slim AS builder

# Install system dependencies for building
# BuildKit cache mounts persist APT cache between builds for faster subsequent builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev

# Install build dependencies
RUN pip install --no-cache-dir poetry-core

# Set up build environment
WORKDIR /build

# Copy project files required for installation
COPY pyproject.toml README.md ./
COPY src ./src

# Install dependencies and project
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
# BuildKit cache mounts persist APT cache between builds for faster subsequent builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    libpq5 \
    curl

# Create non-root user for security
RUN groupadd -r omninode && useradd -r -g omninode omninode

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy source code with proper ownership
COPY --chown=omninode:omninode src/omninode_bridge ./src/omninode_bridge
COPY --chown=omninode:omninode alembic ./alembic
COPY --chown=omninode:omninode alembic.ini ./alembic.ini

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Performance optimizations for BLAKE3
ENV PYTHONHASHSEED=0
ENV BLAKE3_THREADS=4

# Security settings
ENV PYTHONDONTWRITEBYTECODE=1
USER omninode

# Health check - uses simple /health endpoint for fast response
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8053/health || exit 1

# Expose ports
EXPOSE 8053 9090

# Start command
CMD ["python", "-m", "uvicorn", "omninode_bridge.services.metadata_stamping.main:app", \
     "--host", "0.0.0.0", "--port", "8053", "--workers", "1", \
     "--access-log", "--log-level", "info"]
