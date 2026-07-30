# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT

ARG UV_VERSION=0.11.8

FROM postgres:16-alpine AS postgres
COPY docker/application-acl-proof/seed.sql /docker-entrypoint-initdb.d/00-acl-seed.sql

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM python:3.12-slim AS proof

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_HTTP_TIMEOUT=600

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential git libpq-dev postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /usr/local/bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY tests/fixtures/application_database_acl/ ./proof/fixtures/
COPY scripts/ci/prove_application_database_acl.py ./proof/prove.py
COPY docker/application-acl-proof/generated/prechange-fixture-acl.json ./proof/prechange-fixture-acl.json

CMD ["/app/.venv/bin/python", "/app/proof/prove.py"]
