#!/bin/bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-20834 AC2 readback: apply the shipped generation_events migrations on an
# isolated PostgreSQL 16 container and assert that unpriced and omitted costs
# are NULL while existing zero and priced costs keep their values. Needs Docker
# only; no lane credentials or remote database.
#
# Readiness is checked over TCP. The postgres image's first boot runs a
# temporary server on the unix socket only, then restarts it; a socket
# pg_isready answers during that window and the next psql hits the restart
# ("connection to server on socket ... failed"). The init server does not
# listen on TCP, so a TCP answer is the real server. A server that never
# answers fails the run instead of falling through to psql.
set -euo pipefail

cd "$(dirname "$0")/../.."
migrations=docker/migrations/forward/nodes/node_projection_delegation

name="omn20834-readback-$$"
docker run --detach --name "$name" -e POSTGRES_HOST_AUTH_METHOD=trust postgres:16-alpine >/dev/null
trap 'docker rm -f "$name" >/dev/null' EXIT

ready=0
for _ in $(seq 1 60); do
  if docker exec "$name" pg_isready -h 127.0.0.1 -U postgres >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "postgres did not accept TCP connections within 60s" >&2
  exit 1
fi

{
  cat "$migrations/0008_generation_events.sql"
  cat <<'SQL'
INSERT INTO generation_events (correlation_id, cost_inference_usd) VALUES ('existing-priced', 1.234567), ('existing-free', 0);
SQL
  cat "$migrations/0063_generation_events_cost_inference_usd_nullable.sql"
  cat <<'SQL'
INSERT INTO generation_events (correlation_id, cost_inference_usd) VALUES ('unpriced', NULL), ('priced', 2.345678);
INSERT INTO generation_events (correlation_id) VALUES ('omitted');
DO $$ BEGIN
  IF (SELECT count(*) FROM generation_events WHERE correlation_id IN ('unpriced', 'omitted') AND cost_inference_usd IS NULL) <> 2 THEN RAISE EXCEPTION 'unpriced cost is not NULL'; END IF;
  IF (SELECT cost_inference_usd FROM generation_events WHERE correlation_id = 'existing-priced') IS DISTINCT FROM 1.234567::numeric THEN RAISE EXCEPTION 'existing priced cost changed'; END IF;
  IF (SELECT cost_inference_usd FROM generation_events WHERE correlation_id = 'existing-free') IS DISTINCT FROM 0::numeric THEN RAISE EXCEPTION 'existing zero changed'; END IF;
  IF (SELECT cost_inference_usd FROM generation_events WHERE correlation_id = 'priced') IS DISTINCT FROM 2.345678::numeric THEN RAISE EXCEPTION 'priced cost changed'; END IF;
END $$;
SELECT correlation_id, cost_inference_usd FROM generation_events ORDER BY correlation_id;
SQL
} | docker exec -i "$name" psql -X -h 127.0.0.1 -U postgres -v ON_ERROR_STOP=1
