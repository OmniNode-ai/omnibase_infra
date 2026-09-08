#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# run-cloud-migrations.sh -- apply the omninode_cloud forward corpus to the
# compose dev/lab lane, from zero (OMN-17530, epic; corpus OMN-17537).
#
# WHAT THIS IS NOT. It is not a second specification of the apply order. The
# order and the per-entry apply conditions are `MANIFEST`, authored in
# omninode_infra `db/migrations/` and shipped INSIDE the migrate image beside
# the SQL (docker/Dockerfile.migrate), and they are evaluated here by that
# image's OWN `manifest_lib.sh` -- sourced, never reimplemented. This runner
# supplies the connection, the ordering-free bookkeeping and the loud failures;
# it decides nothing about what runs when.
#
# That is deliberate and it is the whole reason this file is short. The corpus
# has four known dependency pairs that a lexicographic sort gets wrong from an
# empty database (20260226_s2 references a column added by 20260226_s4;
# 20260427_backfill references a column added by 20260427_tenant_billing_plan_
# columns; and the sort dies outright at 20260130_create_app_users.sql, 12 of 43
# applied). A compose lane that re-derived an order would be a THIRD
# specification and would drift from the two that exist the first time a
# migration lands.
#
# WHERE IT RUNS. The `cloud-migration` one-shot in the dev-lane compose overlay,
# on the postgres:16-alpine image, after `cloud-migration-files` has copied the
# migrate image's /migrations into the shared volume. That two-step shape mirrors
# the k8s Job's initContainer + container exactly, for the same reason: the SQL
# ships in an alpine image with no psql, and psql ships in an image with no SQL.
#
# WHAT IT DELIBERATELY DOES NOT COPY from k8s/migrations/omninode-cloud-migrate.
# yaml: the 300-second advisory-lock ceremony and the 30-attempt readiness probe.
# Both exist because that Job can be started concurrently by two deploys against
# a shared RDS instance. This one-shot is serialised by compose `depends_on:
# service_completed_successfully` against a single-lane Postgres that is already
# `service_healthy` before it starts, so the lock would be a copy of a block
# `scripts/check-migration-runner-schema-consistency.py` requires to stay
# byte-identical across the k8s runners -- a copy in a file that gate does not
# scan is worse than not having it. If this lane ever gains a second concurrent
# writer, the lock belongs here by EXTRACTION from that manifest, not by a
# fourth transcription.
set -euo pipefail
export PSQL_HISTORY=/dev/null

MIGRATION_DIR="${MIGRATION_DIR:-/migrations}"
DB_HOST="${DB_HOST:?DB_HOST must be set}"
DB_PORT="${DB_PORT:?DB_PORT must be set}"
DB_NAME="${DB_NAME:?DB_NAME must be set}"
DB_USER="${DB_USER:?DB_USER must be set}"
: "${PGPASSWORD:?PGPASSWORD must be set -- the corpus is applied as its owning login, never as the superuser}"

# The corpus arrives from the cloud-migration-files one-shot through a shared
# volume. `depends_on: service_completed_successfully` orders that in a cold
# bring-up and is precisely what a warm `up -d --no-deps` switches off, so the
# ordering is re-established here rather than assumed. Bounded, and it names
# what it was waiting for -- an unbounded wait on a one-shot that already
# failed is a Job that hangs until its deadline with no diagnosis.
CORPUS_READY_TIMEOUT="${CORPUS_READY_TIMEOUT:-120}"
waited=0
while [ ! -f "${MIGRATION_DIR}/.corpus-ready" ]; do
  if [ "$waited" -ge "$CORPUS_READY_TIMEOUT" ]; then
    echo "FATAL: ${MIGRATION_DIR}/.corpus-ready never appeared after ${CORPUS_READY_TIMEOUT}s --" >&2
    echo "       the cloud-migration-files one-shot did not finish copying the corpus." >&2
    exit 1
  fi
  sleep 2
  waited=$((waited + 2))
done

MANIFEST="${MIGRATION_DIR}/MANIFEST"
LIB="${MIGRATION_DIR}/manifest_lib.sh"
BASELINE="${MIGRATION_DIR}/scripts/00_baseline_schema.sql"

# Fail closed, naming the artifact, in all three directions. A missing MANIFEST
# is the one case where a silent fallback would be catastrophic and invisible:
# the run would succeed with a sorted order and leave a database that looks
# migrated and is not.
[ -f "$MANIFEST" ] || {
  echo "FATAL: no ${MANIFEST} in the migrate image. This runner has no lexicographic" >&2
  echo "       fallback by design -- an image built before omninode_infra's MANIFEST" >&2
  echo "       layer cannot be applied from zero. Rebuild the migrate image." >&2
  exit 1
}
[ -f "$LIB" ] || {
  echo "FATAL: no ${LIB} in the migrate image -- the manifest evaluator ships beside" >&2
  echo "       the MANIFEST (omninode_infra docker/Dockerfile.migrate). Rebuild it." >&2
  exit 1
}
# The corpus opens with `ALTER TABLE public.tenants`; the original CREATE TABLE
# is not in it. db/migrations/scripts/00_baseline_schema.sql is the repo's own
# answer (OMN-10545) and .github/workflows/onex-api-tests.yml already runs it
# first against its fresh CI Postgres. It is NOT matched by Dockerfile.migrate's
# `COPY db/migrations/*.sql` glob (it lives in a subdirectory), so it must be
# added to that image explicitly. Named here rather than substituted, because a
# baseline this lane authored itself would be a second copy of a schema
# omninode_infra owns.
[ -f "$BASELINE" ] || {
  echo "FATAL: no ${BASELINE} in the migrate image." >&2
  echo "       The omninode_cloud corpus cannot be applied to an EMPTY database without" >&2
  echo "       it: 20251207_tenants_uuid_pk.sql opens with 'ALTER TABLE public.tenants'." >&2
  echo "       omninode_infra docker/Dockerfile.migrate must also carry" >&2
  echo "       'COPY db/migrations/scripts/00_baseline_schema.sql /migrations/scripts/'." >&2
  exit 1
}

# shellcheck source=/dev/null
. "$LIB"

psql_db() {
  psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 "$@"
}
psql_server() {
  psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -v ON_ERROR_STOP=1 "$@"
}

echo "== omninode_cloud migration (compose dev lane) =="
echo "   target ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Every failed probe prints the server's own text. The pre-OMN-15272 shape of
# this loop in the k8s runner sent stderr to /dev/null after stdout, which
# discarded the one fact that distinguishes "postgres is still starting" from
# "postgres actively refused THIS role" -- and refusing this role is the
# expected first failure here, because role_omninode does not exist on a lane
# whose ROLE_OMNINODE_PASSWORD was empty at initdb time.
DB_READY_ATTEMPTS="${DB_READY_ATTEMPTS:-30}"
attempt=0
while ! probe_error="$(psql_db -c 'SELECT 1' 2>&1 >/dev/null)"; do
  attempt=$((attempt + 1))
  # The password never reaches psql's argv (PGPASSWORD is in the environment)
  # and is scrubbed from the server text defensively in case a future edit
  # changes that.
  echo "waiting for postgres (${attempt}/${DB_READY_ATTEMPTS}): ${probe_error//$PGPASSWORD/***}"
  if [ "$attempt" -ge "$DB_READY_ATTEMPTS" ]; then
    echo "FATAL: ${DB_USER}@${DB_NAME} not reachable after ${DB_READY_ATTEMPTS} attempts." >&2
    echo "FATAL: last psql error: ${probe_error//$PGPASSWORD/***}" >&2
    echo "       If this is 'role \"${DB_USER}\" does not exist', the lane was initialised" >&2
    echo "       with an empty ROLE_OMNINODE_PASSWORD -- 000_create_multiple_databases.sh" >&2
    echo "       treats empty as skip. Run scripts/runtime_build/render_dev_lane_tenant_path_env.sh" >&2
    echo "       and re-run that init script against the existing volume." >&2
    exit 1
  fi
  sleep 2
done

# The manifest's `requires_database` condition is evaluated against the set of
# databases that actually exist on THIS server, read live -- not against a list.
# manifest_verdict takes them comma-separated.
EXISTING_DATABASES="$(psql_server -tAc 'SELECT datname FROM pg_database WHERE NOT datistemplate' | paste -sd, -)"
echo "   databases on this server: ${EXISTING_DATABASES}"

manifest_assert_complete "$MANIFEST" "$MIGRATION_DIR"
echo "   manifest and corpus agree"

echo "-- baseline: ${BASELINE}"
psql_db -f "$BASELINE"

# The corpus's own tracking table ships as 00000000_migrations_tracking.sql, the
# manifest's first entry. This bootstrap row exists only so the ALREADY-APPLIED
# probe below has a table to read on the very first pass; it is created with the
# same shape the corpus creates and the corpus's own CREATE is idempotent.
psql_db -c "CREATE TABLE IF NOT EXISTS public.schema_migrations (
              migration_name text PRIMARY KEY,
              applied_at timestamptz NOT NULL DEFAULT now()
            )"

applied=0
skipped=0
already=0
while IFS="$(printf '\t')" read -r name conditions; do
  [ -n "$name" ] || continue

  if [ "$(psql_db -tAc "SELECT count(*) FROM public.schema_migrations WHERE migration_name = '${name}'")" != "0" ]; then
    echo "-- ALREADY APPLIED ${name}"
    already=$((already + 1))
    continue
  fi

  verdict="$(manifest_verdict "$conditions" "$EXISTING_DATABASES")"
  case "$verdict" in
    SKIP\ *)
      # Loud, and it names the file and the reason. A silent skip here is the
      # OMN-16026 defect class -- a migration that merges and is applied nowhere.
      echo "-- SKIPPED ${name}: ${verdict#SKIP }"
      skipped=$((skipped + 1))
      continue
      ;;
    FATAL\ *)
      echo "FATAL: ${name}: ${verdict#FATAL }" >&2
      exit 1
      ;;
    APPLY) ;;
    *)
      echo "FATAL: ${name}: manifest_verdict returned an unrecognised verdict '${verdict}'" >&2
      exit 1
      ;;
  esac

  echo "-- APPLY ${name} ${conditions}"
  # GUC values reach psql on STDIN, never argv, so they cannot appear in a
  # process list. manifest_guc_prelude is empty for an entry with no GUCs.
  { manifest_guc_prelude "$conditions"; cat "${MIGRATION_DIR}/${name}"; } | psql_db -f -
  psql_db -c "INSERT INTO public.schema_migrations (migration_name) VALUES ('${name}')
              ON CONFLICT (migration_name) DO NOTHING"
  applied=$((applied + 1))
done <<EOF
$(manifest_entries "$MANIFEST")
EOF

echo "== omninode_cloud migration complete: applied=${applied} already=${already} skipped=${skipped} =="
