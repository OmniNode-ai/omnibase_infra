#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# run-forward-migrations.sh — Apply omnibase_infra forward migrations on warm Postgres volumes
#
# Tracks applied migrations in public.schema_migrations and applies any
# pending files from /migrations/forward in sorted order. Safe to run on
# both fresh volumes (no-op for files already applied via docker-entrypoint-initdb.d)
# and warm volumes (applies any new files not yet recorded).
#
# This script is run by the forward-migration compose service before migration-gate
# checks the sentinel flag. It replaces the docker-entrypoint-initdb.d-only
# mechanism for keeping warm Postgres volumes up-to-date with new migrations.
#
# Ticket: OMN-4175 (Forward migration runner for warm Postgres volumes)
# Ticket: OMN-13062 (migration-gate vacuity fix — retro A-10)
#
# ---------------------------------------------------------------------------
# Sentinel discipline (OMN-13062)
# ---------------------------------------------------------------------------
# migrations_complete is cleared to FALSE at the start of every runner
# invocation and set to TRUE only as the FINAL act after all infra and
# node migrations apply without error. Any nonzero exit from any migration
# leaves the flag FALSE, making the migration-gate healthcheck UNHEALTHY.
#
# The committed per-migration skip-manifest is the SOLE escape for migrations
# that must be intentionally skipped:
#   docker/migrations/skip-manifest.yaml
# Format:
#   skipped_migrations:
#     - id: "docker/NNN_name.sql"
#       reason: "..."
#       ticket: "OMN-XXXX"
# The runner reads this at startup; a listed migration_id is treated as
# already-applied without executing the SQL. New entries must be committed
# in the same PR that deems the migration unrunnable.
#
# ---------------------------------------------------------------------------
# Node-owned migration auto-discovery (OMN-12559)
# ---------------------------------------------------------------------------
# omnimarket projection nodes ship SQL under
# src/omnimarket/nodes/<node>/migrations/*.sql. Those files are vendored into
# this repo under ${MIGRATIONS_DIR}/nodes/<node>/ (kept in sync by
# scripts/sync-node-migrations.sh) so a clean clone reproduces the views with
# NO manual copy and NO manual renumber.
#
# Each node migration is tracked under a NAMESPACED migration_id of the form
#   node:<node>:<filename>
# This is a separate identity space from the flat infra sequence
# (tracked as docker/<filename>). Because the namespace is distinct, a node
# migration numbered e.g. 076 NEVER collides with infra's flat 076 file —
# the renumber-as-operational-pattern is eliminated.
#
# Environment:
#   POSTGRES_USER     (default: postgres)
#   POSTGRES_PASSWORD (required)
#   POSTGRES_HOST     (default: postgres)
#   POSTGRES_PORT     (default: 5432)
#   POSTGRES_DB       (default: omnibase_infra)
#   MIGRATIONS_DIR    (default: /migrations/forward)
#   NODE_MIGRATIONS_DIR (default: ${MIGRATIONS_DIR}/nodes)
#   NODE_POSTGRES_DB  (default: POSTGRES_DB; compose sets omnidash_analytics)
#   PG_WAIT_RETRIES   (default: 30 — number of 2s waits for postgres ready)
#   FORWARD_MIGRATION_LOCK_ID      (default: 100010 — advisory lock id, OMN-15291)
#   MIGRATION_LOCK_WAIT_SECONDS    (default: 300 — bounded wait for that lock)

set -e

PGUSER="${POSTGRES_USER:-postgres}"
PGHOST="${POSTGRES_HOST:-postgres}"
PGPORT="${POSTGRES_PORT:-5432}"
PGDB="${POSTGRES_DB:-omnibase_infra}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations/forward}"
NODE_MIGRATIONS_DIR="${NODE_MIGRATIONS_DIR:-${MIGRATIONS_DIR}/nodes}"
NODE_PGDB="${NODE_POSTGRES_DB:-${PGDB}}"
PG_WAIT_RETRIES="${PG_WAIT_RETRIES:-30}"

export PGPASSWORD="${POSTGRES_PASSWORD}"

# ---------------------------------------------------------------------------
# Skip-manifest: load intentionally-skipped migration ids (OMN-13062)
# ---------------------------------------------------------------------------
# Format: YAML file with a top-level list "skipped_migrations" each entry has
# "id" (e.g. "docker/038_placeholder.sql") and optionally "reason" / "ticket".
# Only a committed manifest is honoured — operator env cannot inject skips.
SKIP_MANIFEST="${MIGRATIONS_DIR%/forward}/skip-manifest.yaml"
SKIPPED_IDS=""
if [ -f "${SKIP_MANIFEST}" ]; then
  echo "[forward-migration] Loading skip-manifest: ${SKIP_MANIFEST}"
  # Extract quoted id: values from YAML using portable sed (no yq/python/gawk).
  # Handles lines of the form:  - id: "docker/NNN_name.sql"
  SKIPPED_IDS="$(sed -n 's/^[[:space:]]*-[[:space:]]*id:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "${SKIP_MANIFEST}" 2>/dev/null || true)"
fi

is_skipped_by_manifest() {
  migration_id="$1"
  if [ -z "${SKIPPED_IDS}" ]; then
    return 1
  fi
  echo "${SKIPPED_IDS}" | grep -Fxq "${migration_id}"
}

# ---------------------------------------------------------------------------
# 0. Wait for Postgres to be ready (first-boot initdb race guard, OMN-13062)
# ---------------------------------------------------------------------------
echo "[forward-migration] Waiting for Postgres to accept connections..."
retries=0
until psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -c "SELECT 1" >/dev/null 2>&1; do
  retries=$((retries + 1))
  if [ "$retries" -ge "$PG_WAIT_RETRIES" ]; then
    echo "[forward-migration] ERROR: Postgres not ready after ${PG_WAIT_RETRIES} retries. Aborting." >&2
    exit 1
  fi
  echo "[forward-migration]   postgres not ready (attempt ${retries}/${PG_WAIT_RETRIES}), retrying in 2s..."
  sleep 2
done
echo "[forward-migration] Postgres is ready."

# ---- BEGIN canonical forward-migration advisory lock (OMN-15291) ----
# Port of the OMN-15254 single-session lock to this runner, in POSIX sh (this
# script is #!/bin/sh and runs under busybox ash in the migration container --
# no `local`, no arrays, no bashisms).
#
# What was wrong: this runner had NO lock of any kind. Every migration was
# applied through an unsynchronized check-then-act (SELECT from
# schema_migrations -> psql -f -> INSERT ... ON CONFLICT DO NOTHING), so two
# concurrent runners both read "not applied" and both executed the same file.
# Non-idempotent DDL then errored in the loser, and the ON CONFLICT hid the
# double-apply so schema_migrations still looked clean afterwards.
#
# Why this shape: pg_advisory_lock() is SESSION-scoped. Acquiring it with a
# one-shot `psql -c` releases it the instant psql exits, and a later
# pg_advisory_unlock() from a different session returns false without ever
# having held anything. The lock is therefore held by ONE dedicated psql
# session that stays alive for the whole run; release is that session ending,
# never a cross-session unlock.
#
# Scope: advisory locks are per-DATABASE. This lock is taken in ${PGDB} and
# held across BOTH the infra phase (${PGDB}) and the node phase (${NODE_PGDB}),
# so two instances of THIS runner are fully serialized against each other. It
# does not serialize against unrelated writers of ${NODE_PGDB}.
#
# Lock id 100010 is deliberately outside omninode_infra's k8s Job registry
# (100001-100006, see omninode_infra/scripts/run-migrations.sh) so the two
# runners never contend on a shared id by accident.
FORWARD_MIGRATION_LOCK_ID="${FORWARD_MIGRATION_LOCK_ID:-100010}"
MIGRATION_LOCK_WAIT_SECONDS="${MIGRATION_LOCK_WAIT_SECONDS:-300}"
MIGRATION_LOCK_TAG="forward-migration-lock-${FORWARD_MIGRATION_LOCK_ID}-$$"
MIGRATION_LOCK_OWNER_PID=""

# True only when the lock is granted to OUR holder session. Matching on
# application_name is what makes this specific: "somebody holds it" is not
# proof that WE hold it. A psql error is reported as not-held.
migration_lock_held() {
  _mlh_granted="$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -tAc \
    "SELECT count(*) FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid
       WHERE l.locktype = 'advisory' AND l.granted AND l.objsubid = 1
         AND l.classid::bigint * 4294967296 + l.objid::bigint = ${FORWARD_MIGRATION_LOCK_ID}
         AND a.application_name = '${MIGRATION_LOCK_TAG}';" 2>/dev/null)" || return 1
  [ "$_mlh_granted" = "1" ]
}

# Release by ending the holder session: Postgres drops session-level advisory
# locks on disconnect. No cross-session unlock call anywhere. Confirmation
# polls the lock itself instead of using `wait`: the holder is a background
# PIPELINE, and `wait` on it blocks until every member exits -- a deadlock
# when this runs from the EXIT trap.
release_migration_lock() {
  [ -n "$MIGRATION_LOCK_OWNER_PID" ] || return 0
  kill "$MIGRATION_LOCK_OWNER_PID" 2>/dev/null || true
  MIGRATION_LOCK_OWNER_PID=""
  _rml_waited=0
  while migration_lock_held && [ "$_rml_waited" -lt 10 ]; do
    sleep 1
    _rml_waited=$((_rml_waited + 1))
  done
}

acquire_migration_lock() {
  echo "[forward-migration] Acquiring advisory lock ${FORWARD_MIGRATION_LOCK_ID} (single session, bounded wait ${MIGRATION_LOCK_WAIT_SECONDS}s)..."
  {
    # statement_timeout bounds the blocking acquire IN THE SERVER, so a
    # contended lock fails loud instead of hanging until the deploy times out.
    echo "SET statement_timeout = '${MIGRATION_LOCK_WAIT_SECONDS}s';"
    echo "SELECT pg_advisory_lock(${FORWARD_MIGRATION_LOCK_ID});"
    echo "SET statement_timeout = 0;"
    # Hold this session's stdin -- and therefore the session, and therefore the
    # lock -- open for the rest of the run. The payload is a SQL comment: psql
    # reads and discards it with no server round trip, but the write fails with
    # EPIPE the moment the holder session is gone, which is how this loop
    # learns to stop. The tick cap is a backstop so an orphaned keepalive can
    # never outlive the day.
    _aml_ticks=0
    while [ "$_aml_ticks" -lt 86400 ]; do
      sleep 1
      echo "-- forward migration advisory lock keepalive" || break
      _aml_ticks=$((_aml_ticks + 1))
    done
  # 2>/dev/null is load-bearing: this group must not inherit the runner's
  # stderr. It can outlive the runner by up to a second, and any parent reading
  # the runner's output to EOF would otherwise block on the inherited pipe.
  } 2>/dev/null | PGAPPNAME="$MIGRATION_LOCK_TAG" psql -h "$PGHOST" -p "$PGPORT" \
        -U "$PGUSER" -d "$PGDB" -v ON_ERROR_STOP=1 -q -o /dev/null &
  MIGRATION_LOCK_OWNER_PID=$!

  _aml_deadline=$(( $(date +%s) + MIGRATION_LOCK_WAIT_SECONDS + 5 ))
  until migration_lock_held; do
    if ! kill -0 "$MIGRATION_LOCK_OWNER_PID" 2>/dev/null; then
      MIGRATION_LOCK_OWNER_PID=""
      echo "[forward-migration] FATAL: could not acquire advisory lock ${FORWARD_MIGRATION_LOCK_ID} -- holder session exited before the lock was granted (another forward-migration run likely holds it, or the ${MIGRATION_LOCK_WAIT_SECONDS}s statement_timeout fired)" >&2
      exit 1
    fi
    if [ "$(date +%s)" -ge "$_aml_deadline" ]; then
      echo "[forward-migration] FATAL: could not acquire advisory lock ${FORWARD_MIGRATION_LOCK_ID} within ${MIGRATION_LOCK_WAIT_SECONDS}s -- refusing to apply migrations unserialized" >&2
      release_migration_lock
      exit 1
    fi
    sleep 1
  done
  echo "[forward-migration] Advisory lock ${FORWARD_MIGRATION_LOCK_ID} acquired; held by session '${MIGRATION_LOCK_TAG}' for the whole run."
}

# Called as the last act before the sentinel is set TRUE: a holder session that
# died mid-run means the migrations above may have run unserialized, so
# reporting success (and flipping the migration gate HEALTHY) would be a lie.
assert_migration_lock_still_held() {
  if ! kill -0 "$MIGRATION_LOCK_OWNER_PID" 2>/dev/null || ! migration_lock_held; then
    echo "[forward-migration] FATAL: advisory lock ${FORWARD_MIGRATION_LOCK_ID} was NOT held for the whole run -- the holder session died mid-run and migrations above may have run unserialized" >&2
    exit 1
  fi
}

# Traps set BEFORE acquisition so a partially-started holder is still reaped on
# set -e failures and signals.
#
# EXIT and the signals are deliberately SEPARATE traps. In POSIX sh only the
# EXIT trap is terminal: a HUP/INT/TERM handler that returns normally RESUMES
# the script, so a single combined trap would release the lock mid-run and then
# keep applying migrations unserialized until the final held-ness assertion
# noticed. The signal handler therefore exits non-zero itself; the EXIT trap
# then re-runs release_migration_lock, which is idempotent (it returns
# immediately once MIGRATION_LOCK_OWNER_PID is cleared).
trap 'release_migration_lock' EXIT
trap 'release_migration_lock; echo "[forward-migration] FATAL: terminated by signal before completion" >&2; exit 1' HUP INT TERM
acquire_migration_lock
# ---- END canonical forward-migration advisory lock (OMN-15291) ----

validate_database_identifier() {
  database="$1"
  if ! printf '%s' "$database" | grep -Eq '^[A-Za-z_][A-Za-z0-9_-]*$'; then
    echo "[forward-migration] invalid database identifier in migration directive: ${database}" >&2
    exit 1
  fi
}

ensure_directive_database() {
  migration_file="$1"
  directive_line="$(
    grep -i -m 1 -E '^--[[:space:]]*onex-create-database[[:space:]]*:' "$migration_file" || true
  )"
  if [ -z "$directive_line" ]; then
    return 0
  fi

  database="$(
    printf '%s\n' "$directive_line" \
      | sed -E 's/^--[[:space:]]*onex-create-database[[:space:]]*:[[:space:]]*//; s/[[:space:]]*$//'
  )"
  validate_database_identifier "$database"
  echo "[forward-migration]   ensure database ${database}..."
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -v ON_ERROR_STOP=1 <<-EOSQL
    SELECT 'CREATE DATABASE "$database"'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
EOSQL
}

# ---------------------------------------------------------------------------
# 1. Ensure schema_migrations tracking table exists (idempotent)
# ---------------------------------------------------------------------------
echo "[forward-migration] Ensuring schema_migrations table exists..."

psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -c "
CREATE TABLE IF NOT EXISTS public.schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum     TEXT NOT NULL,
    source_set   TEXT NOT NULL
);
"

# ---------------------------------------------------------------------------
# 1a. Clear the sentinel at the start of every run (OMN-13062)
# ---------------------------------------------------------------------------
# This ensures that any mid-run failure leaves migrations_complete=FALSE so
# the migration-gate healthcheck stays UNHEALTHY. The sentinel is only set
# TRUE as the very last act of this script (after all migrations succeed).
# We use a conditional UPDATE so this is a no-op on volumes that have not
# yet applied migration 037 (migrations_complete column may not exist yet).
echo "[forward-migration] Clearing migration sentinel (will be re-set on successful completion)..."
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -c "
DO \$\$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'db_metadata'
      AND column_name = 'migrations_complete'
  ) THEN
    UPDATE public.db_metadata
    SET migrations_complete = FALSE,
        updated_at = NOW()
    WHERE id = TRUE;
  END IF;
END;
\$\$;
" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Apply pending migrations in sorted order
# ---------------------------------------------------------------------------
echo "[forward-migration] Scanning ${MIGRATIONS_DIR} for pending migrations..."

APPLIED=0
SKIPPED=0

for migration_file in $(ls "${MIGRATIONS_DIR}"/*.sql | sort); do
  filename=$(basename "$migration_file")
  migration_id="docker/${filename}"

  # Honour skip-manifest: treat manifest-listed migrations as already applied
  if is_skipped_by_manifest "${migration_id}"; then
    echo "[forward-migration]   skip  ${filename} (skip-manifest)"
    SKIPPED=$((SKIPPED + 1))
    # Record in schema_migrations so the table stays consistent.
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
      -c "INSERT INTO public.schema_migrations (migration_id, checksum, source_set)
          VALUES ('${migration_id}', 'skip-manifest', 'docker')
          ON CONFLICT (migration_id) DO NOTHING;"
    continue
  fi

  # Check if already applied
  already_applied=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
    -tAc "SELECT 1 FROM public.schema_migrations WHERE migration_id = '${migration_id}'" 2>/dev/null || true)

  if [ "$already_applied" = "1" ]; then
    echo "[forward-migration]   skip  ${filename} (already applied)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "[forward-migration]   apply ${filename}..."

  # Apply migration then record in tracking table
  ensure_directive_database "$migration_file"
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
    -v ON_ERROR_STOP=1 -f "$migration_file"

  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
    -c "INSERT INTO public.schema_migrations (migration_id, checksum, source_set)
        VALUES ('${migration_id}', 'applied-by-runner', 'docker')
        ON CONFLICT (migration_id) DO NOTHING;"

  echo "[forward-migration]   done  ${filename}"
  APPLIED=$((APPLIED + 1))
done

# ---------------------------------------------------------------------------
# 3. Auto-discover and apply node-owned migrations (OMN-12559)
# ---------------------------------------------------------------------------
# Walk ${NODE_MIGRATIONS_DIR}/<node>/*.sql. Within each node directory files
# are applied in sorted (lexical) order. Each file is tracked under the
# namespaced id  node:<node>:<filename>  so the infra numeric sequence is
# never collided with and no renumber is ever required.
NODE_APPLIED=0
NODE_SKIPPED=0

if [ -d "${NODE_MIGRATIONS_DIR}" ]; then
  echo "[forward-migration] Ensuring schema_migrations table exists in node projection database ${NODE_PGDB}..."

  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" -c "
CREATE TABLE IF NOT EXISTS public.schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum     TEXT NOT NULL,
    source_set   TEXT NOT NULL
);
"

  echo "[forward-migration] Scanning ${NODE_MIGRATIONS_DIR} for node-owned migrations in ${NODE_PGDB}..."

  # Iterate node directories in sorted order for deterministic application.
  for node_dir in $(ls -d "${NODE_MIGRATIONS_DIR}"/*/ 2>/dev/null | sort); do
    node_name=$(basename "$node_dir")

    # Skip directories with no .sql files.
    if ! ls "${node_dir}"*.sql >/dev/null 2>&1; then
      continue
    fi

    for migration_file in $(ls "${node_dir}"*.sql | sort); do
      filename=$(basename "$migration_file")
      migration_id="node:${node_name}:${filename}"

      already_applied=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" \
        -tAc "SELECT 1 FROM public.schema_migrations WHERE migration_id = '${migration_id}'" 2>/dev/null || true)

      if [ "$already_applied" = "1" ]; then
        echo "[forward-migration]   skip  ${migration_id} (already applied)"
        NODE_SKIPPED=$((NODE_SKIPPED + 1))
        continue
      fi

      echo "[forward-migration]   apply ${migration_id}..."

      psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" \
        -v ON_ERROR_STOP=1 -f "$migration_file"

      psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" \
        -c "INSERT INTO public.schema_migrations (migration_id, checksum, source_set)
            VALUES ('${migration_id}', 'applied-by-runner', 'node')
            ON CONFLICT (migration_id) DO NOTHING;"

      echo "[forward-migration]   done  ${migration_id}"
      NODE_APPLIED=$((NODE_APPLIED + 1))
    done
  done
else
  echo "[forward-migration] No node migrations dir at ${NODE_MIGRATIONS_DIR} — skipping node discovery."
fi

echo "[forward-migration] Complete: ${APPLIED} infra applied, ${SKIPPED} infra skipped; ${NODE_APPLIED} node applied, ${NODE_SKIPPED} node skipped."

# ---------------------------------------------------------------------------
# 4. Set the sentinel TRUE only after ALL migrations succeed (OMN-13062)
# ---------------------------------------------------------------------------
# This is the FINAL act. Any earlier failure leaves migrations_complete=FALSE.
# runner_completed_at records the timestamp of this successful completion.
#
# The lock must still be ours at this point (OMN-15291): flipping the gate
# HEALTHY after a run that may have been unserialized is exactly the false
# green this lock exists to prevent.
assert_migration_lock_still_held
echo "[forward-migration] All migrations complete. Setting sentinel TRUE..."
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" -v ON_ERROR_STOP=1 -c "
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    runner_completed_at = NOW(),
    updated_at = NOW()
WHERE id = TRUE;
"
echo "[forward-migration] Sentinel set. Migration gate will report HEALTHY."
