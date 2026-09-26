# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT
#
# OMN-15356 rebuilt PostgreSQL 16 fixture for the capability_scores
# tenant_id TEXT -> UUID conversion. The postgres stage lands the stream up to
# the pre-conversion state (0001, 0002); the proof stage carries the
# conversion (0003) and the two follow-on files in the same stream (0004 x2)
# and applies them itself, so each phase can observe the table before and
# after. Everything runs in a disposable tmpfs container; nothing here reaches
# a live, shared or deployed database.

FROM postgres:16-alpine AS postgres
COPY docker/tenant-uuid-conversion-proof/seed-roles.sql /docker-entrypoint-initdb.d/00-seed-roles.sql
COPY docker/migrations/forward/nodes/node_canary_score_reducer/0001_create_capability_scores.sql /docker-entrypoint-initdb.d/01-create-capability-scores.sql
COPY docker/migrations/forward/nodes/node_canary_score_reducer/0002_capability_scores_tenant_id_and_rls.sql /docker-entrypoint-initdb.d/02-tenant-id-and-rls.sql

FROM python:3.12-slim AS proof
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir psycopg2-binary==2.9.10

WORKDIR /app
COPY scripts/ci/prove_tenant_uuid_conversion.py ./prove.py
COPY docker/migrations/forward/nodes/node_canary_score_reducer/0003_capability_scores_tenant_id_to_uuid.sql \
     docker/migrations/forward/nodes/node_canary_score_reducer/0004_capability_scores_policy_atomic_restatement.sql \
     docker/migrations/forward/nodes/node_canary_score_reducer/0004_grant_tenant_projection_writer_capability_scores_sequence.sql \
     ./migrations/

CMD ["python", "/app/prove.py"]
