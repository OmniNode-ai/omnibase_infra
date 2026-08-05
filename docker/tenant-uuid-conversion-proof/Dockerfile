# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT

FROM postgres:16-alpine AS postgres
COPY docker/tenant-uuid-conversion-proof/seed-roles.sql /docker-entrypoint-initdb.d/00-seed-roles.sql
COPY docker/migrations/forward/nodes/node_canary_score_reducer/0001_create_capability_scores.sql /docker-entrypoint-initdb.d/01-create-capability-scores.sql
COPY docker/migrations/forward/nodes/node_canary_score_reducer/0002_capability_scores_tenant_id_and_rls.sql /docker-entrypoint-initdb.d/02-tenant-id-and-rls.sql

FROM python:3.12-slim AS proof
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psycopg2-binary==2.9.10

WORKDIR /app
COPY scripts/ci/prove_tenant_uuid_conversion.py ./prove.py
COPY docker/migrations/forward/nodes/node_canary_score_reducer/0003_capability_scores_tenant_id_to_uuid.sql \
     ./0003_capability_scores_tenant_id_to_uuid.sql

CMD ["python", "/app/prove.py"]
