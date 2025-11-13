# Dockerfile for OmniNode Bridge Model Metrics API Service
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including snappy for Kafka
# BuildKit cache mounts persist APT cache between builds for faster subsequent builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libsnappy-dev

# Install Poetry with pinned version
RUN pip install poetry==2.1.3

# Configure Poetry: Don't create virtual environment, install dependencies globally
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
# BuildKit cache mount persists Poetry cache between builds (wheels, metadata) for faster installs
RUN --mount=type=cache,target=/tmp/poetry_cache \
    poetry config virtualenvs.create false \
    && poetry install --without dev --no-root

# Copy source code
COPY src/ ./src/

# Create logs directory
RUN mkdir -p /app/logs

# Create a non-root user for security
RUN groupadd -r metricsuser && useradd -r -g metricsuser -d /app -s /bin/bash metricsuser
RUN chown -R metricsuser:metricsuser /app

# Switch to non-root user
USER metricsuser

# Set Python path to include src
ENV PYTHONPATH=/app/src

# Expose the service port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8005/ || exit 1

# Run the Model Metrics API service
CMD ["python", "-m", "omninode_bridge.model_metrics_main"]
