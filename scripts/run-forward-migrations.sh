#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# run-forward-migrations.sh — Apply omnibase_infra forward migrations on warm Postgres volumes
#
# Tracks service-owned flat migrations in public.schema_migrations and
# application/node migrations in platform_catalog.schema_migrations. Applies
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
# ---------------------------------------------------------------------------
# Operator fence for node migrations (OMN-15336 — parity with the k8s runner)
# ---------------------------------------------------------------------------
# The node-migration id space above is SHARED with omninode_infra's k8s Job
# runner (k8s/migrations/omnibase-infra-migrate.yaml): both walk the same
# vendored nodes/<node>/*.sql tree and both mint the id
# node:<node>:<filename>. Until OMN-15336 only the k8s runner carried the
# operator fence, so the two sanctioned paths disagreed about what is gated and
# every compose lane applied the gated migrations unattended. See
# FENCED_NODE_MIGRATION_IDS below for the semantics and the seam.
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
#   ONEX_MIGRATION_LANE (default: unset = FULL operator fence, OMN-15379).
#     Lane indicator for the lane-scoped fence release. The ONLY recognised
#     value is `dev` (the lab compose lane); unset or unknown means every fenced
#     node migration is skipped. Set by docker/docker-compose.dev-lane.yml,
#     which only the dev/lab project loads — deliberately NOT set in
#     docker-compose.infra.yml, which every lane overlay merges. See the
#     LANE-SCOPED FENCE RELEASE block below.

set -e

PGUSER="${POSTGRES_USER:-postgres}"
PGHOST="${POSTGRES_HOST:-postgres}"
PGPORT="${POSTGRES_PORT:-5432}"
PGDB="${POSTGRES_DB:-omnibase_infra}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations/forward}"
NODE_MIGRATIONS_DIR="${NODE_MIGRATIONS_DIR:-${MIGRATIONS_DIR}/nodes}"
NODE_PGDB="${NODE_POSTGRES_DB:-${PGDB}}"
PG_WAIT_RETRIES="${PG_WAIT_RETRIES:-30}"
LEDGER_BOOTSTRAP="${MIGRATIONS_DIR}/_ledger/bootstrap.sql"
APPLICATION_MIGRATION_MANIFEST="${MIGRATIONS_DIR}/_ledger/application-migrations.tsv"
APPLICATION_MIGRATION_BLOCKS="${MIGRATIONS_DIR}/_ledger/application-migration-blocks.tsv"
LEGACY_NODE_MIGRATION_DECLARATIONS="${MIGRATIONS_DIR}/_ledger/legacy-node-migrations.tsv"
CLOUD_MIGRATION_ALIASES="${MIGRATIONS_DIR}/_ledger/cloud-migration-aliases.tsv"

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

# ---- BEGIN operator fence — node migration ids (OMN-15336) ----
# SINGLE-SOURCED from docker/migrations/forward/fenced-node-migrations.yaml
# (OMN-15349). That file is the baseline operator fence over the id space
# this runner and omninode_infra/k8s/migrations/omnibase-infra-migrate.yaml
# share (node:<node>:<filename>, minted over the same vendored SQL tree) — a
# fence in only one of them is not a fence: OMN-15336 found the .201 dev lane
# and the stability-test lane had applied all of the gated ids and were
# running FORCE ROW LEVEL SECURITY on six tables, while the cloud RDS copy
# the k8s runner drives was clean.
#
# Before OMN-15349 this list was a hand-maintained literal here AND in the
# k8s manifest — two copies that had already drifted once (k8s durably
# released the registration trio under operator ruling 21 while this runner
# stayed at the full baseline; see the LANE-SCOPED FENCE RELEASE block below
# for why that is a *different* release mechanism, not a parity break).
# Loading the same manifest file both runners read removes that drift class
# for the baseline; each runner's release policy on top of the baseline is
# still its own, because the release is an environment-specific operator
# decision, not fence data.
#
# Loaded from the manifest, NOT `${FENCED_NODE_MIGRATION_IDS:-...}`: only a
# COMMITTED manifest is honoured, exactly as with the skip-manifest above. An
# operator env var must not be able to supply or empty this list — see
# test_fence_is_not_overridable_by_environment.
FENCE_MANIFEST="${MIGRATIONS_DIR}/fenced-node-migrations.yaml"
if [ ! -f "${FENCE_MANIFEST}" ]; then
  echo "FATAL: operator fence manifest not found: ${FENCE_MANIFEST}" >&2
  exit 1
fi
FENCED_NODE_MIGRATION_IDS="$(sed -n \
  's/^[[:space:]]*-[[:space:]]*id:[[:space:]]*"\([^"]*\)".*/\1/p' \
  "${FENCE_MANIFEST}")"
if [ -z "${FENCED_NODE_MIGRATION_IDS}" ]; then
  # Not FATAL: an empty fence is a legitimate future state (every id
  # eventually released) as well as the symptom of a malformed manifest, and
  # this sed grammar cannot tell the two apart. Blocking the whole migration
  # run on an empty fence would make "no ids currently need gating" a worse
  # outage than the fence gap it replaces. The committed-content checks
  # (test_manifest_pins_the_known_baseline_fence,
  # test_manifest_shell_parse_matches_yaml_parse) are what catch an
  # accidentally-emptied manifest, at PR time, before it ships.
  echo "[forward-migration] WARNING: operator fence manifest ${FENCE_MANIFEST}" \
    "parsed to an empty list — no node migrations are currently fenced" >&2
fi

is_fenced_node_migration() {
  candidate="$1"
  printf '%s\n' "${FENCED_NODE_MIGRATION_IDS}" | grep -Fxq "${candidate}"
}

# --- LANE-SCOPED FENCE RELEASE (OMN-15379 — operator ruling 15, 2026-07-29) ---
# Operator ruling 15: node_service_registry FORCE ROW LEVEL SECURITY extends to
# the LAB LANE ONLY. The lab (compose dev lane, project `omnibase-infra`) applies
# the registration trio IN FULL — CREATE + heartbeat columns + ENABLE/FORCE RLS —
# as the proving ground that generates the evidence the staging un-fence is
# waiting on.
#
# CORRECTION (OMN-15349, 2026-08-05): this comment previously claimed "the
# staging k8s fence is UNCHANGED and stays at all seven ids." That was true
# when ruling 15 landed (2026-07-29) and stale within two days: operator
# ruling 21 (OMN-15332 comment 1a067542, 2026-07-31T14:05Z GO) durably
# released the registration trio on the k8s/staging side too — permanently,
# not env-gated like this lab-lane release, because that Job serves exactly
# one environment. So THIS runner's release (below) and the k8s runner's
# release are two independently-ruled policies over the same shared baseline
# manifest (docker/migrations/forward/fenced-node-migrations.yaml), not one
# parity relationship — do not re-derive "the two runners must show the same
# effective fence" from this block; they intentionally do not right now.
#
# FAIL-CLOSED BY CONSTRUCTION. Three independent properties, each tested:
#
#   1. DEFAULT IS FULLY FENCED. ${ONEX_MIGRATION_LANE} unset -> empty release set
#      -> every one of the seven ids is skipped, exactly as before this change.
#      An UNKNOWN value is treated the same as unset (and warns). There is no
#      value that widens the fence relative to today; a lane can only ever
#      release a SUBSET of it.
#   2. THE RELEASE SET IS COMMITTED, NOT SUPPLIED. The env var selects among
#      policies that are literal in this file; it never carries ids. No env
#      value can release an id that is not written below, and the release is
#      only ever consulted for an id that is already fenced (see the node loop),
#      so a lane cannot "release" something the fence does not cover.
#   3. NOT INHERITABLE. The indicator is NOT set in docker-compose.infra.yml.
#      Every non-dev lane overlay (stability-test / prod / judge) MERGES that
#      base file, so anything set there would be inherited by all of them —
#      fail-OPEN, and silently so for any lane added later. It is instead set by
#      docker/docker-compose.dev-lane.yml, an overlay that ONLY the dev/lab
#      project loads. A lane that does not load that file — stability-test,
#      prod, judge, CI, a raw `docker compose -f docker-compose.infra.yml up` on
#      a fresh volume, and any future lane — gets no indicator and the full
#      fence. Asserted over the RENDERED compose config in
#      tests/scripts/test_node_migration_fence_parity.py.
#
# HONEST LIMIT: a deliberate `-e ONEX_MIGRATION_LANE=dev` on another lane's
# forward-migration container would release the trio there. That is an explicit
# operator act on the same footing as editing the fence list itself, and it is
# not defended against — the container has no un-forgeable lane fact available
# to corroborate against (compose does not inject the project name, and the
# service's own container_name is not readable from inside it).
#
# Delegation 0023-0026 are NOT releasable on ANY lane. Ruling 15 is scoped to
# node_service_registry; the delegation tenant-RLS hold is a separate ruling
# still pending, and no case arm below names those ids.
ONEX_MIGRATION_LANE="${ONEX_MIGRATION_LANE:-}"
case "${ONEX_MIGRATION_LANE}" in
  dev)
    # The lab/dev compose lane. Ruling 15 proving ground.
    LANE_RELEASED_NODE_MIGRATION_IDS="\
node:node_projection_registration:0000_create_node_service_registry.sql
node:node_projection_registration:0001_add_heartbeat_columns.sql
node:node_projection_registration:0002_node_service_registry_tenant_rls.sql"
    ;;
  "")
    LANE_RELEASED_NODE_MIGRATION_IDS=""
    ;;
  *)
    echo "[forward-migration] WARNING: unknown ONEX_MIGRATION_LANE='${ONEX_MIGRATION_LANE}' — applying the FULL operator fence (fail-closed)." >&2
    LANE_RELEASED_NODE_MIGRATION_IDS=""
    ;;
esac

is_lane_released_node_migration() {
  candidate="$1"
  if [ -z "${LANE_RELEASED_NODE_MIGRATION_IDS}" ]; then
    return 1
  fi
  printf '%s\n' "${LANE_RELEASED_NODE_MIGRATION_IDS}" | grep -Fxq "${candidate}"
}
# ---- END operator fence — node migration ids (OMN-15336) ----

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
# Canonical application migration ledger (OMN-15413)
# ---------------------------------------------------------------------------
# The approved topology contract selects platform_catalog.schema_migrations
# with explicit stream/domain/version/checksum columns.  bootstrap.sql evolves
# the checksum-capable node relation in place before any already-applied probe;
# the filename-only relation remains immutable import evidence.

validate_migration_identity() {
  identity="$1"
  if ! printf '%s' "$identity" | grep -Eq '^[A-Za-z0-9_./:-]+$'; then
    echo "[forward-migration] FATAL: invalid migration identity '${identity}'" >&2
    exit 1
  fi
}

validate_client_file_path() {
  client_file_path="$1"
  case "$client_file_path" in
    ""|*[!A-Za-z0-9_./-]*)
      echo "[forward-migration] FATAL: unsafe psql client file path '${client_file_path}'" >&2
      exit 1
      ;;
  esac
  if [ ! -f "$client_file_path" ]; then
    echo "[forward-migration] FATAL: psql client file is missing: ${client_file_path}" >&2
    exit 1
  fi
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

validate_application_migration_manifest() {
  for manifest_file in \
    "$APPLICATION_MIGRATION_MANIFEST" \
    "$APPLICATION_MIGRATION_BLOCKS" \
    "$LEGACY_NODE_MIGRATION_DECLARATIONS" \
    "$CLOUD_MIGRATION_ALIASES"
  do
    if [ ! -f "$manifest_file" ]; then
      echo "[forward-migration] FATAL: application migration declaration missing: ${manifest_file}" >&2
      exit 1
    fi
  done

  if ! awk -F '\t' '
    NF != 6 { exit 1 }
    {
      path_count = split($1, path_parts, "/")
      expected_owner = "node:" path_parts[2]
      expected_version = expected_owner ":" path_parts[3]
    }
    path_count != 3 || path_parts[1] != "nodes" { exit 1 }
    path_parts[2] !~ /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/ { exit 1 }
    path_parts[3] !~ /^[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql$/ { exit 1 }
    $2 != expected_owner || $3 != expected_owner || $5 != expected_version { exit 1 }
    $4 != "tenant" && $4 != "omninode_internal" { exit 1 }
    $6 !~ /^[0-9a-f]{64}$/ { exit 1 }
  ' "$APPLICATION_MIGRATION_MANIFEST"; then
    echo "[forward-migration] FATAL: malformed or unknown stream/owner/domain in application migration manifest" >&2
    exit 1
  fi
  if ! awk -F '\t' '
    NF != 5 { exit 1 }
    {
      path_count = split($1, path_parts, "/")
      expected_version = "node:" path_parts[2] ":" path_parts[3]
    }
    path_count != 3 || path_parts[1] != "nodes" { exit 1 }
    path_parts[2] !~ /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/ { exit 1 }
    path_parts[3] !~ /^[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql$/ { exit 1 }
    $2 != expected_version || $3 !~ /^[0-9a-f]{64}$/ { exit 1 }
    $4 !~ /^OMN-[0-9]+$/ || $5 == "" { exit 1 }
  ' "$APPLICATION_MIGRATION_BLOCKS"; then
    echo "[forward-migration] FATAL: malformed application migration block declaration" >&2
    exit 1
  fi
  if ! awk -F '\t' '
    NF != 6 { exit 1 }
    {
      version_count = split($4, version_parts, ":")
      expected_owner = "node:" version_parts[2]
    }
    version_count != 3 || version_parts[1] != "node" { exit 1 }
    version_parts[2] !~ /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/ { exit 1 }
    version_parts[3] !~ /^[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql$/ { exit 1 }
    $1 != expected_owner || $2 != expected_owner { exit 1 }
    $3 != "tenant" && $3 != "omninode_internal" { exit 1 }
    $5 !~ /^[A-Za-z0-9_.:-]+$/ || $6 !~ /^OMN-[0-9]+$/ { exit 1 }
  ' "$LEGACY_NODE_MIGRATION_DECLARATIONS"; then
    echo "[forward-migration] FATAL: malformed historical node migration declaration" >&2
    exit 1
  fi
  if ! awk -F '\t' 'NF != 2 || $1 !~ /^[A-Za-z0-9_.-]+$/ || $2 !~ /^[A-Za-z0-9_.-]+[.]sql$/ { exit 1 }' \
    "$CLOUD_MIGRATION_ALIASES"; then
    echo "[forward-migration] FATAL: malformed cloud migration alias declaration" >&2
    exit 1
  fi
  if [ -n "$(cut -f 1 "$CLOUD_MIGRATION_ALIASES" | sort | uniq -d | head -n 1)" ] \
     || [ -n "$(cut -f 2 "$CLOUD_MIGRATION_ALIASES" | sort | uniq -d | head -n 1)" ]; then
    echo "[forward-migration] FATAL: duplicate cloud migration alias declaration" >&2
    exit 1
  fi

  duplicate_artifact="$(cut -f 1 "$APPLICATION_MIGRATION_MANIFEST" | sort | uniq -d | head -n 1)"
  duplicate_identity="$(cut -f 2,4,5 "$APPLICATION_MIGRATION_MANIFEST" | sort | uniq -d | head -n 1)"
  if [ -n "$duplicate_artifact" ]; then
    echo "[forward-migration] FATAL: double migration declaration for artifact ${duplicate_artifact}" >&2
    exit 1
  fi
  if [ -n "$duplicate_identity" ]; then
    echo "[forward-migration] FATAL: duplicate migration version ${duplicate_identity}" >&2
    exit 1
  fi
  duplicate_legacy_version="$(cut -f 4 "$LEGACY_NODE_MIGRATION_DECLARATIONS" | sort | uniq -d | head -n 1)"
  legacy_active_overlap="$( { cut -f 5 "$APPLICATION_MIGRATION_MANIFEST"; cut -f 2 "$APPLICATION_MIGRATION_BLOCKS"; cut -f 4 "$LEGACY_NODE_MIGRATION_DECLARATIONS"; } | sort | uniq -d | head -n 1)"
  if [ -n "$duplicate_legacy_version" ] || [ -n "$legacy_active_overlap" ]; then
    echo "[forward-migration] FATAL: ambiguous historical node migration declaration" >&2
    exit 1
  fi

  while IFS='	' read -r legacy_stream legacy_owner _ legacy_version _ _; do
    legacy_node="$(printf '%s' "$legacy_version" | cut -d ':' -f 2)"
    legacy_filename="$(printf '%s' "$legacy_version" | cut -d ':' -f 3)"
    if [ -f "${NODE_MIGRATIONS_DIR}/${legacy_node}/${legacy_filename}" ]; then
      echo "[forward-migration] FATAL: historical declaration has active artifact ${legacy_version}" >&2
      exit 1
    fi
  done <"$LEGACY_NODE_MIGRATION_DECLARATIONS"

  while IFS='	' read -r artifact_path _ _ _ declared_version declared_checksum; do
    migration_file="${MIGRATIONS_DIR}/${artifact_path}"
    if [ ! -f "$migration_file" ]; then
      echo "[forward-migration] FATAL: declared migration artifact missing: ${artifact_path}" >&2
      exit 1
    fi
    actual_checksum="$(file_sha256 "$migration_file")"
    if [ "$actual_checksum" != "$declared_checksum" ]; then
      echo "[forward-migration] FATAL: conflicting migration checksum for ${declared_version}" >&2
      exit 1
    fi
  done <"$APPLICATION_MIGRATION_MANIFEST"

  while IFS='	' read -r artifact_path blocked_version blocked_checksum blocked_ticket blocked_reason; do
    migration_file="${MIGRATIONS_DIR}/${artifact_path}"
    if [ ! -f "$migration_file" ] || [ "$(file_sha256 "$migration_file")" != "$blocked_checksum" ]; then
      echo "[forward-migration] FATAL: conflicting migration checksum for blocked ${blocked_version}" >&2
      exit 1
    fi
    if ! is_fenced_node_migration "$blocked_version" \
       || is_lane_released_node_migration "$blocked_version"; then
      echo "[forward-migration] FATAL: unresolved migration domain for ${blocked_version} (${blocked_ticket}: ${blocked_reason})" >&2
      exit 1
    fi
  done <"$APPLICATION_MIGRATION_BLOCKS"

  for migration_file in $(find "$NODE_MIGRATIONS_DIR" -mindepth 2 -maxdepth 2 -type f -name '*.sql' | sort); do
    artifact_path="nodes/${migration_file#"${NODE_MIGRATIONS_DIR}"/}"
    declared_count="$(awk -F '\t' -v path="$artifact_path" '$1 == path { count++ } END { print count + 0 }' \
      "$APPLICATION_MIGRATION_MANIFEST")"
    blocked_count="$(awk -F '\t' -v path="$artifact_path" '$1 == path { count++ } END { print count + 0 }' \
      "$APPLICATION_MIGRATION_BLOCKS")"
    if [ $((declared_count + blocked_count)) -ne 1 ]; then
      echo "[forward-migration] FATAL: migration ${artifact_path} must have exactly one declaration or explicit block" >&2
      exit 1
    fi
  done
}

resolve_application_migration() {
  artifact_path="$1"
  expected_version="$2"
  declaration="$(awk -F '\t' -v path="$artifact_path" '$1 == path { print }' \
    "$APPLICATION_MIGRATION_MANIFEST")"
  if [ -z "$declaration" ]; then
    block="$(awk -F '\t' -v path="$artifact_path" '$1 == path { print }' \
      "$APPLICATION_MIGRATION_BLOCKS")"
    if [ -n "$block" ]; then
      block_ticket="$(printf '%s\n' "$block" | cut -f 4)"
      block_reason="$(printf '%s\n' "$block" | cut -f 5)"
      echo "[forward-migration] FATAL: unresolved migration domain for ${expected_version} (${block_ticket}: ${block_reason})" >&2
    else
      echo "[forward-migration] FATAL: unknown application migration ${artifact_path}" >&2
    fi
    exit 1
  fi

  DECLARED_STREAM="$(printf '%s\n' "$declaration" | cut -f 2)"
  DECLARED_OWNER="$(printf '%s\n' "$declaration" | cut -f 3)"
  DECLARED_DOMAIN="$(printf '%s\n' "$declaration" | cut -f 4)"
  DECLARED_VERSION="$(printf '%s\n' "$declaration" | cut -f 5)"
  DECLARED_CHECKSUM="$(printf '%s\n' "$declaration" | cut -f 6)"
  if [ "$DECLARED_VERSION" != "$expected_version" ]; then
    echo "[forward-migration] FATAL: migration version mismatch for ${artifact_path}" >&2
    exit 1
  fi
}

prepare_canonical_ledger() {
  ledger_database="$1"
  validate_database_identifier "$ledger_database"
  if [ ! -f "$LEDGER_BOOTSTRAP" ]; then
    echo "[forward-migration] FATAL: canonical ledger bootstrap missing: ${LEDGER_BOOTSTRAP}" >&2
    exit 1
  fi
  validate_client_file_path "$APPLICATION_MIGRATION_MANIFEST"
  validate_client_file_path "$LEGACY_NODE_MIGRATION_DECLARATIONS"
  echo "[forward-migration] Converging canonical ledger in ${ledger_database}..."
  psql -X -q -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$ledger_database" \
    -v ON_ERROR_STOP=1 \
    -c "CREATE TEMP TABLE onex_application_migration_manifest (
          artifact_path TEXT NOT NULL,
          migration_stream TEXT NOT NULL,
          owner TEXT NOT NULL,
          domain TEXT NOT NULL,
          version TEXT NOT NULL,
          checksum TEXT NOT NULL,
          PRIMARY KEY (artifact_path),
          UNIQUE (migration_stream, domain, version)
        ); CREATE TEMP TABLE onex_legacy_node_migration_declarations (
          migration_stream TEXT NOT NULL,
          owner TEXT NOT NULL,
          domain TEXT NOT NULL,
          version TEXT NOT NULL PRIMARY KEY,
          source_checksum TEXT NOT NULL,
          ticket TEXT NOT NULL
        )" \
    -c "\copy onex_application_migration_manifest FROM '${APPLICATION_MIGRATION_MANIFEST}' WITH (FORMAT text, DELIMITER E'\t')" \
    -c "\copy onex_legacy_node_migration_declarations FROM '${LEGACY_NODE_MIGRATION_DECLARATIONS}' WITH (FORMAT text, DELIMITER E'\t')" \
    -f "$LEDGER_BOOTSTRAP"
}

# Return 0 only when the version is already present and its canonical metadata
# is safe to skip.  Content hashes must match byte-for-byte.  A
# legacy_attestation row is deliberately distinguishable and can never satisfy
# an active node migration probe: it proves a source record, not file bytes.
migration_is_applied() {
  ledger_database="$1"
  ledger_stream="$2"
  ledger_owner="$3"
  ledger_domain="$4"
  ledger_version="$5"
  expected_checksum="$6"
  validate_migration_identity "$ledger_version"

  ledger_row="$(
    psql -X -qAt -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$ledger_database" \
      -v ON_ERROR_STOP=1 \
      -v ledger_stream="$ledger_stream" \
      -v ledger_version="$ledger_version" \
      -v ledger_domain="$ledger_domain" \
      -f - <<'EOSQL'
SELECT checksum || '|' || checksum_kind || '|' || owner || '|' || provenance
          FROM platform_catalog.schema_migrations
          WHERE migration_stream = :'ledger_stream'
            AND domain = :'ledger_domain'
            AND version = :'ledger_version';
EOSQL
  )"
  if [ -z "$ledger_row" ]; then
    return 1
  fi

  recorded_checksum="$(printf '%s\n' "$ledger_row" | cut -d '|' -f 1)"
  recorded_kind="$(printf '%s\n' "$ledger_row" | cut -d '|' -f 2)"
  recorded_owner="$(printf '%s\n' "$ledger_row" | cut -d '|' -f 3)"

  if [ "$recorded_owner" != "$ledger_owner" ]; then
    echo "[forward-migration] FATAL: double migration declaration for ${ledger_stream}:${ledger_domain}:${ledger_version} (recorded owner ${recorded_owner}, declared ${ledger_owner})" >&2
    exit 1
  fi
  case "$recorded_kind" in
    content_sha256)
      if [ "$recorded_checksum" != "$expected_checksum" ]; then
        echo "[forward-migration] FATAL: conflicting migration checksum for ${ledger_stream}:${ledger_domain}:${ledger_version}" >&2
        exit 1
      fi
      ;;
    legacy_attestation)
      echo "[forward-migration] FATAL: active migration ${ledger_version} has only a legacy checksum attestation" >&2
      exit 1
      ;;
    *)
      echo "[forward-migration] FATAL: unknown checksum kind '${recorded_kind}' for ${ledger_version}" >&2
      exit 1
      ;;
  esac
  return 0
}

record_migration() {
  ledger_database="$1"
  ledger_stream="$2"
  ledger_owner="$3"
  ledger_domain="$4"
  ledger_version="$5"
  ledger_checksum="$6"
  ledger_provenance="$7"
  validate_migration_identity "$ledger_version"
  if ! printf '%s' "$ledger_checksum" | grep -Eq '^[0-9a-f]{64}$'; then
    echo "[forward-migration] FATAL: malformed SHA-256 for ${ledger_version}" >&2
    exit 1
  fi

  psql -X -q -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$ledger_database" \
    -v ON_ERROR_STOP=1 \
    -v ledger_stream="$ledger_stream" \
    -v ledger_owner="$ledger_owner" \
    -v ledger_domain="$ledger_domain" \
    -v ledger_version="$ledger_version" \
    -v ledger_checksum="$ledger_checksum" \
    -v ledger_provenance="$ledger_provenance" \
    -f - <<'EOSQL'
INSERT INTO platform_catalog.schema_migrations (
          migration_stream, owner, domain, version, checksum, checksum_kind, provenance
        ) VALUES (
          :'ledger_stream', :'ledger_owner', :'ledger_domain', :'ledger_version',
          :'ledger_checksum', 'content_sha256', :'ledger_provenance'
        );
EOSQL
}

database_exists() {
  candidate_database="$1"
  validate_database_identifier "$candidate_database"
  [ "$(
    psql -X -qAt -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
      -v ON_ERROR_STOP=1 -v candidate_database="$candidate_database" \
      -f - <<'EOSQL'
SELECT count(*) FROM pg_database WHERE datname = :'candidate_database';
EOSQL
  )" = "1" ]
}

import_ledger_stage() {
  target_database="$1"
  stage_file="$2"
  if [ ! -s "$stage_file" ]; then
    return 0
  fi
  validate_client_file_path "$stage_file"

  psql -X -q -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$target_database" \
    -v ON_ERROR_STOP=1 \
    -c "BEGIN; CREATE TEMP TABLE onex_migration_import_stage (
          migration_stream TEXT NOT NULL,
          owner TEXT NOT NULL,
          domain TEXT NOT NULL,
          version TEXT NOT NULL,
          checksum TEXT NOT NULL,
          checksum_kind TEXT NOT NULL,
          applied_at TIMESTAMPTZ NOT NULL,
          provenance TEXT NOT NULL
        ) ON COMMIT DROP" \
    -c "\copy onex_migration_import_stage FROM '${stage_file}' WITH (FORMAT csv)" \
    -f - <<'EOSQL'
DO $import_validation$
BEGIN
  IF EXISTS (
    SELECT 1 FROM onex_migration_import_stage
    WHERE NOT (
      migration_stream = 'omninode-cloud'
      AND owner = 'service:onex_api'
      AND domain = 'legacy_unclassified'
    )
  ) THEN
    RAISE EXCEPTION 'unknown migration stream/domain declaration in import';
  END IF;
  IF EXISTS (
    SELECT 1 FROM onex_migration_import_stage
    WHERE checksum !~ '^[0-9a-f]{64}$'
       OR checksum_kind NOT IN ('content_sha256', 'legacy_attestation')
  ) THEN
    RAISE EXCEPTION 'malformed migration checksum in import';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM onex_migration_import_stage
    GROUP BY migration_stream, domain, version
    HAVING count(*) > 1
  ) THEN
    RAISE EXCEPTION 'duplicate migration version in import';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM onex_migration_import_stage incoming
    JOIN platform_catalog.schema_migrations existing
      USING (migration_stream, domain, version)
    WHERE incoming.checksum <> existing.checksum
  ) THEN
    RAISE EXCEPTION 'conflicting migration checksum in import';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM onex_migration_import_stage incoming
    JOIN platform_catalog.schema_migrations existing
      USING (migration_stream, domain, version)
    WHERE incoming.checksum = existing.checksum
      AND (
        incoming.owner <> existing.owner
        OR incoming.checksum_kind <> existing.checksum_kind
        OR incoming.applied_at <> existing.applied_at
        OR incoming.provenance <> existing.provenance
      )
  ) THEN
    RAISE EXCEPTION 'double migration declaration in import';
  END IF;
END
$import_validation$;

INSERT INTO platform_catalog.schema_migrations (
  migration_stream, owner, domain, version, checksum, checksum_kind,
  applied_at, provenance
)
SELECT incoming.migration_stream, incoming.owner, incoming.domain, incoming.version,
       incoming.checksum, incoming.checksum_kind, incoming.applied_at,
       incoming.provenance
FROM onex_migration_import_stage incoming
WHERE NOT EXISTS (
  SELECT 1
  FROM platform_catalog.schema_migrations existing
  WHERE existing.migration_stream = incoming.migration_stream
    AND existing.domain = incoming.domain
    AND existing.version = incoming.version
);
COMMIT;
EOSQL
}

import_cloud_history() {
  target_database="$1"
  cloud_database="${OMNINODE_CLOUD_HISTORY_DB:-omninode_cloud}"
  validate_database_identifier "$cloud_database"
  if ! database_exists "$cloud_database"; then
    echo "[forward-migration] Historical cloud database ${cloud_database} absent; no cloud history to import."
    return 0
  fi

  stage_file="$(mktemp)"
  validate_client_file_path "$CLOUD_MIGRATION_ALIASES"
  psql -X -q -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$cloud_database" \
    -v ON_ERROR_STOP=1 \
    -c "CREATE TEMP TABLE onex_cloud_migration_alias (
          migration_name TEXT PRIMARY KEY,
          runner_version TEXT NOT NULL UNIQUE
        )" \
    -c "CREATE TEMP TABLE onex_cloud_migration_export (
          migration_stream TEXT NOT NULL,
          owner TEXT NOT NULL,
          domain TEXT NOT NULL,
          version TEXT NOT NULL,
          checksum TEXT NOT NULL,
          checksum_kind TEXT NOT NULL,
          applied_at TIMESTAMPTZ NOT NULL,
          provenance TEXT NOT NULL
        )" \
    -c "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY" \
    -c "\copy onex_cloud_migration_alias FROM '${CLOUD_MIGRATION_ALIASES}' WITH (FORMAT text, DELIMITER E'\t')" \
    -f - <<'EOSQL' >"$stage_file"
DO $cloud_history_export$
DECLARE
  schema_columns TEXT;
  log_columns TEXT;
BEGIN
  SELECT coalesce(string_agg(column_name, ',' ORDER BY column_name), '')
  INTO schema_columns
  FROM information_schema.columns
  WHERE table_schema = 'public' AND table_name = 'schema_migrations';
  SELECT coalesce(string_agg(column_name, ',' ORDER BY column_name), '')
  INTO log_columns
  FROM information_schema.columns
  WHERE table_schema = 'public' AND table_name = 'migrations_log';

  IF schema_columns = '' THEN
    IF log_columns <> '' THEN
      RAISE EXCEPTION
        'cloud migrations_log is audit-only: applied-set ledger is absent';
    END IF;
    RETURN;
  END IF;
  IF schema_columns <> 'applied_at,checksum,version' OR EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'schema_migrations'
      AND (
        (column_name = 'version'
          AND (udt_name <> 'text' OR is_nullable <> 'NO'))
        OR (column_name = 'checksum' AND udt_name <> 'text')
        OR (column_name = 'applied_at'
          AND (udt_name <> 'timestamptz' OR is_nullable <> 'NO'))
      )
  ) THEN
    RAISE EXCEPTION
      'unknown cloud applied-set ledger shape: public.schema_migrations (%)',
      schema_columns;
  END IF;

  IF log_columns <> '' AND (
    NOT ('migration_name' = ANY (string_to_array(log_columns, ',')))
    OR NOT ('direction' = ANY (string_to_array(log_columns, ',')))
    OR NOT ('executed_at' = ANY (string_to_array(log_columns, ',')))
    OR EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'migrations_log'
        AND (
          (column_name IN ('migration_name', 'direction')
            AND (udt_name <> 'text' OR is_nullable <> 'NO'))
          OR (column_name = 'executed_at'
            AND (udt_name <> 'timestamptz' OR is_nullable <> 'NO'))
        )
    )
  ) THEN
    RAISE EXCEPTION
      'unknown cloud audit ledger shape: public.migrations_log (%)', log_columns;
  END IF;

  IF log_columns = '' THEN
    INSERT INTO onex_cloud_migration_export
    SELECT
      'omninode-cloud',
      'service:onex_api',
      'legacy_unclassified',
      applied.version,
      CASE
        WHEN applied.checksum ~ '^[0-9a-f]{64}$' THEN applied.checksum
        ELSE encode(sha256(convert_to(
          'omninode-cloud|legacy_unclassified|' || applied.version || '|' ||
          coalesce(applied.checksum, '<NULL>'), 'UTF8'
        )), 'hex')
      END,
      CASE
        WHEN applied.checksum ~ '^[0-9a-f]{64}$' THEN 'content_sha256'
        ELSE 'legacy_attestation'
      END,
      applied.applied_at,
      format(
        'legacy:%s:public.schema_migrations:version:%s:raw-checksum=%s',
        current_database(), applied.version, coalesce(applied.checksum, '<NULL>')
      )
    FROM public.schema_migrations applied
    ORDER BY applied.version;
    RETURN;
  END IF;

  IF EXISTS (
    SELECT 1 FROM public.migrations_log
    WHERE direction IS NULL OR direction NOT IN ('forward', 'rollback')
  ) THEN
    RAISE EXCEPTION 'unknown cloud migrations_log direction';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM public.migrations_log log
    LEFT JOIN onex_cloud_migration_alias alias
      ON alias.migration_name = log.migration_name
    WHERE alias.migration_name IS NULL
  ) THEN
    RAISE EXCEPTION 'unknown cloud migrations_log alias';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM public.migrations_log log
    JOIN onex_cloud_migration_alias alias
      ON alias.migration_name = log.migration_name
    LEFT JOIN public.schema_migrations applied
      ON applied.version = alias.runner_version
    WHERE applied.version IS NULL
  ) THEN
    RAISE EXCEPTION
      'cloud migrations_log is audit-only: log-only alias cannot be imported as applied';
  END IF;

  INSERT INTO onex_cloud_migration_export
  SELECT
    'omninode-cloud',
    'service:onex_api',
    'legacy_unclassified',
    applied.version,
    CASE
      WHEN applied.checksum ~ '^[0-9a-f]{64}$' THEN applied.checksum
      ELSE encode(sha256(convert_to(
        'omninode-cloud|legacy_unclassified|' || applied.version || '|' ||
        coalesce(applied.checksum, '<NULL>'), 'UTF8'
      )), 'hex')
    END,
    CASE
      WHEN applied.checksum ~ '^[0-9a-f]{64}$' THEN 'content_sha256'
      ELSE 'legacy_attestation'
    END,
    applied.applied_at,
    format(
      'legacy:%s:public.schema_migrations:version:%s:raw-checksum=%s%s',
      current_database(), applied.version, coalesce(applied.checksum, '<NULL>'),
      CASE WHEN bool_or(log.migration_name IS NOT NULL)
        THEN ';migrations_log:' || max(log.migration_name)
        ELSE ''
      END
    )
  FROM public.schema_migrations applied
  LEFT JOIN onex_cloud_migration_alias alias
    ON alias.runner_version = applied.version
  LEFT JOIN public.migrations_log log
    ON log.migration_name = alias.migration_name
  GROUP BY applied.version, applied.checksum, applied.applied_at
  ORDER BY applied.version;
END
$cloud_history_export$;

COPY onex_cloud_migration_export TO STDOUT WITH (FORMAT csv);
COMMIT;
EOSQL
  import_ledger_stage "$target_database" "$stage_file"
  rm -f "$stage_file"
}

# Validate the complete checked-in application declaration surface before
# either database is mutated. Service-only invocations with no node tree keep
# using only the separate flat ledger and do not bootstrap an application DB.
if [ -d "$NODE_MIGRATIONS_DIR" ]; then
  validate_application_migration_manifest
fi

# ---------------------------------------------------------------------------
# 1. Ensure service-owned schema_migrations tracking table exists (idempotent)
# ---------------------------------------------------------------------------
echo "[forward-migration] Ensuring service migration ledger exists in ${PGDB}..."

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
    # Record in the service-owned ledger so the table stays consistent.
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
      -c "INSERT INTO public.schema_migrations (migration_id, checksum, source_set)
          VALUES ('${migration_id}', 'skip-manifest', 'docker')
          ON CONFLICT (migration_id) DO NOTHING;"
    continue
  fi

  # The flat set belongs to the separate omnibase_infra service database.  Its
  # runner/ledger remain out of the unified application-database scope.
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

# Converge only the unified application database when this invocation actually
# carries the node migration tree. omnibase_infra remains a separate
# service-owned database under plan section 0.1.
if [ -d "$NODE_MIGRATIONS_DIR" ]; then
  prepare_canonical_ledger "$NODE_PGDB"
  import_cloud_history "$NODE_PGDB"
fi

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
      migration_checksum="$(file_sha256 "$migration_file")"

      # ---- BEGIN fenced-id skip (OMN-15336) ----
      # FIRST, ahead of the ledger probe and the apply. Skip and count it;
      # never apply it, never record it in schema_migrations.
      #
      # OMN-15379: the lane-scoped release is checked INSIDE the fenced branch,
      # never beside it. An id that is not fenced never consults the release
      # set, and the release set can therefore only ever un-gate a strict subset
      # of the fence — it can never gate or un-gate anything else.
      if is_fenced_node_migration "${migration_id}"; then
        if is_lane_released_node_migration "${migration_id}"; then
          echo "[forward-migration]   RELEASED on lane '${ONEX_MIGRATION_LANE}' (operator ruling 15, OMN-15379): ${migration_id}"
          # Fall through to the normal already-applied probe + apply + record.
        else
          echo "[forward-migration]   SKIP (operator-gated, see OMN-14974/OMN-15313/OMN-15335/OMN-15343): ${migration_id}"
          NODE_SKIPPED=$((NODE_SKIPPED + 1))
          continue
        fi
      fi
      # ---- END fenced-id skip (OMN-15336) ----

      artifact_path="nodes/${node_name}/${filename}"
      resolve_application_migration "$artifact_path" "$migration_id"
      if [ "$DECLARED_CHECKSUM" != "$migration_checksum" ]; then
        echo "[forward-migration] FATAL: conflicting migration checksum for ${migration_id}" >&2
        exit 1
      fi

      if migration_is_applied \
        "$NODE_PGDB" "$DECLARED_STREAM" "$DECLARED_OWNER" "$DECLARED_DOMAIN" \
        "$migration_id" "$migration_checksum"; then
        echo "[forward-migration]   skip  ${migration_id} (already applied)"
        NODE_SKIPPED=$((NODE_SKIPPED + 1))
        continue
      fi

      echo "[forward-migration]   apply ${migration_id}..."

      psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$NODE_PGDB" \
        -v ON_ERROR_STOP=1 -f "$migration_file"

      record_migration \
        "$NODE_PGDB" "$DECLARED_STREAM" "$DECLARED_OWNER" "$DECLARED_DOMAIN" \
        "$migration_id" "$migration_checksum" \
        "file:nodes/${node_name}/${filename}"

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
