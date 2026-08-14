# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT

ARG UV_VERSION=0.11.8
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_HTTP_TIMEOUT=600

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /usr/local/bin/
WORKDIR /app

# Dependency layer is rebuilt only when the exact lock surface changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the actual runtime source and typed topology fixture after dependencies.
COPY src/ ./src/
COPY tests/fixtures/application_relation_ownership/topology.yaml ./proof/topology.yaml
COPY docker/domain-adapter-proof/prove.py ./proof/prove.py

USER 65534:65534
CMD ["/app/.venv/bin/python", "/app/proof/prove.py"]
