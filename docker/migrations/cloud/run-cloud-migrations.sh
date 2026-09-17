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
# The corpus's 00000000_migrations_tracking.sql creates `schema_migrations`
# UNQUALIFIED, so it lands in whatever the creation namespace resolves to, while
# everything here addresses `public.schema_migrations`. Today those agree only
# because no role or database in this lane sets a search_path and no schema is
# named after the role -- and role_omninode holds CREATE on the database, so a
# schema of its own name is reachable rather than impossible. Pinning it makes
# the two provably the same object instead of coincidentally the same one.
export PGOPTIONS="${PGOPTIONS:+$PGOPTIONS }-c search_path=public"

MIGRATION_DIR="${MIGRATION_DIR:-/migrations}"
DB_HOST="${DB_HOST:?DB_HOST must be set}"
DB_PORT="${DB_PORT:?DB_PORT must be set}"
DB_NAME="${DB_NAME:?DB_NAME must be set}"
DB_USER="${DB_USER:?DB_USER must be set}"
: "${PGPASSWORD:?PGPASSWORD must be set -- the corpus is applied as its mapped service role, never as the superuser}"

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

# ---------------------------------------------------------------------------
# The tracker: declared once, by the corpus, and never here (OMN-18544)
# ---------------------------------------------------------------------------
# This runner used to bootstrap public.schema_migrations itself, ahead of the
# manifest loop, so the ALREADY-APPLIED probe below had a table to read on the
# very first pass. It declared `(migration_name text PRIMARY KEY, applied_at
# timestamptz)`. The corpus declares `(version, applied_at, checksum)` in its
# own 00000000_migrations_tracking.sql. Because this runner always won the
# name, that file's `CREATE TABLE IF NOT EXISTS` was a silent no-op on every
# run and the shape it declares never landed -- so 021_workflow_results.sql,
# which self-registers on `version`, died on `column "version" of relation
# "schema_migrations" does not exist`.
#
# That is OMN-4627 with the two sides swapped, and OMN-4627 was closed by
# reconciling the two literals by hand. The hand reconciliation is exactly what
# did not survive: this runner was written afterwards (OMN-17530) and
# reintroduced the divergence. So the fix is NOT a corrected copy of the
# corpus's shape -- a correct copy is how the defect came back. This runner now
# declares nothing at all about the tracker: it applies the corpus's own
# tracking migration as the bootstrap and reads its bookkeeping column off the
# resulting primary key, so THIS REPO carries no literal for a future edit to
# drift away from.
#
# That is the honest scope and it is narrower than "one declaration anywhere".
# omninode_infra's k8s Job runner (k8s/migrations/omninode-cloud-migrate.yaml)
# declares the same shape for the k3s path, and the corpus's own
# 20260429_plan_entitlements.sql declares a third, conditional shape for a
# local-dev database carrying neither column. Neither is reachable from this
# lane and neither is this runner's to reconcile; what this file guarantees is
# that it adds no fourth, which is exactly what the gate below enforces.
#
# This is not a second specification of the apply order either. The file below
# is the MANIFEST's own first entry; applying it here brings the table into
# existence before the resume probe needs it and reorders nothing. The loop
# still visits it, finds it unrecorded on a fresh lane, re-applies it -- every
# statement in it is idempotent -- and records it.
#
# Gate: tests/unit/db/test_schema_migrations_tracker_shape_agreement_omn18544.py
# (sibling gate on the k8s Job runners: check-migration-runner-schema-consistency.py).
TRACKING="${MIGRATION_DIR}/00000000_migrations_tracking.sql"
[ -f "$TRACKING" ] || {
  echo "FATAL: no ${TRACKING} in the migrate image. It is the corpus's own tracking" >&2
  echo "       migration and this runner's ONLY source for the schema_migrations" >&2
  echo "       shape -- by design it has no bootstrap of its own (OMN-18544)." >&2
  echo "       Rebuild the migrate image from omninode_infra db/migrations." >&2
  exit 1
}

# Convergence for a lane this runner already bootstrapped in the retired shape.
# The .201 dev lane is one: it carries a migration_name-keyed table with rows in
# it. The legacy table is moved ASIDE, never dropped -- the corpus then creates
# the canonical table under the freed name and the recorded rows are copied back
# through the derived key below. Detection keys on the legacy column because the
# canonical shape has no such column; that is what makes it unambiguous.
# Re-entrant: a run that died between the rename and the copy leaves the stash
# in place and the next pass finishes the job rather than losing the rows.
#
# The index rename is not tidiness, it is the difference between a converged
# lane and a silently degraded one. Postgres does NOT rename a table's indexes
# or constraints when the table is renamed, so the stash keeps holding
# `schema_migrations_pkey` and `idx_schema_migrations_applied_at`. Measured on
# postgres:16-alpine: without this loop the corpus's own
# `CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at` matches the name
# the STASH still holds and skips with a NOTICE, and the canonical table's
# primary key lands as `schema_migrations_pkey1`. Dropping the stash below then
# takes the applied_at index with it and the converged lane ends up with no such
# index at all -- permanently, and with nothing failing to say so. The names are
# read off pg_index rather than written down, so this frees whatever the lane
# actually holds instead of the two this runner happens to know about.
psql_db -c "DO \$\$
DECLARE
  legacy_index record;
BEGIN
  -- pg_catalog, not information_schema: that view shows only the columns the
  -- current role holds a privilege on, so a legacy table this role cannot read
  -- would report NO migration_name column and skip the convergence in silence.
  -- pg_attribute is not privilege-filtered and answers the question asked.
  IF to_regclass('public.schema_migrations') IS NOT NULL
     AND EXISTS (SELECT 1 FROM pg_attribute
                  WHERE attrelid = to_regclass('public.schema_migrations')
                    AND attname  = 'migration_name'
                    AND attnum > 0
                    AND NOT attisdropped)
  THEN
    IF to_regclass('public.schema_migrations_legacy_omn18544') IS NOT NULL THEN
      RAISE EXCEPTION 'OMN-18544: a retired ledger stash already exists while public.schema_migrations is still in the retired shape; resolve by hand rather than letting the rename collide';
    END IF;
    RAISE NOTICE 'OMN-18544: retiring the migration_name-keyed ledger, rows preserved';
    ALTER TABLE public.schema_migrations RENAME TO schema_migrations_legacy_omn18544;
    FOR legacy_index IN
      SELECT c.relname
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
       WHERE i.indrelid = 'public.schema_migrations_legacy_omn18544'::regclass
    LOOP
      EXECUTE format('ALTER INDEX public.%I RENAME TO %I',
                     legacy_index.relname,
                     'legacy_omn18544_' || legacy_index.relname);
    END LOOP;
  END IF;
END
\$\$;"

# Applying the tracking file outside the loop bypasses the MANIFEST's per-entry
# apply conditions for that one entry. That is only safe while the entry carries
# none, which is true today -- so it is checked rather than assumed. If
# omninode_infra ever gives it a condition, this runner would silently ignore it
# and the header's "decides nothing about what runs when" would stop being true.
TRACKING_CONDITIONS="$(manifest_entries "$MANIFEST" | awk -F'\t' -v n="00000000_migrations_tracking.sql" '$1 == n { print $2 }')"
if [ -n "$TRACKING_CONDITIONS" ]; then
  echo "FATAL: the MANIFEST now gives 00000000_migrations_tracking.sql the apply" >&2
  echo "       conditions '${TRACKING_CONDITIONS}'. This runner applies it outside" >&2
  echo "       the loop to bootstrap the tracker and would ignore them (OMN-18544)." >&2
  exit 1
fi

echo "-- tracker bootstrap (corpus-owned): 00000000_migrations_tracking.sql"
psql_db -f "$TRACKING"

# Prove the convergence happened. Every step above is conditional and each one
# is SILENT when its condition is false: a skipped rename leaves the retired
# table in place, the corpus's CREATE TABLE IF NOT EXISTS then no-ops against it
# exactly as it did before this fix, and the key read below resolves to
# `migration_name` off the surviving primary key -- announcing the retired key on
# a line that reads like success, and dying 46 files later on the original error.
# Refusing here is the difference between a fix and a fix that cannot tell you it
# did not work.
#
# This is the ONE assertion this runner is entitled to make about the shape. Not
# what the tracker's columns are, which is the corpus's business and the whole
# point of the change above -- only that the column THIS RUNNER retired is gone.
# Two refusals, not one, because they have different causes and different
# remedies. `to_regclass` returns NULL rather than raising, so an absent table is
# a distinguishable answer instead of a psql abort mid-`if` that would print the
# WRONG diagnosis below. PGOPTIONS above pins search_path, so "absent" here means
# the corpus's own CREATE did not run, not that it landed in another schema.
# Captured OUTSIDE the `if`. A bare assignment takes the substitution's status,
# so a psql that failed for any other reason -- a dropped connection, an auth
# refusal -- aborts here under set -e. Inside an `if` condition set -e is
# suspended and the empty result would compare unequal below, printing this
# refusal's diagnosis for a cause it does not describe.
TRACKER_PRESENT="$(psql_db -tAc "SELECT count(*) FROM pg_class WHERE oid = to_regclass('public.schema_migrations')")"
if [ "$TRACKER_PRESENT" != "1" ]; then
  echo "FATAL: the corpus bootstrap left no public.schema_migrations at all." >&2
  echo "       00000000_migrations_tracking.sql applied without creating its own" >&2
  echo "       tracking table, which this runner has no shape of its own to fall" >&2
  echo "       back on by design (OMN-18544)." >&2
  exit 1
fi
RETIRED_COLUMN="$(psql_db -tAc "SELECT count(*) FROM pg_attribute WHERE attrelid = to_regclass('public.schema_migrations') AND attname = 'migration_name' AND attnum > 0 AND NOT attisdropped")"
if [ "$RETIRED_COLUMN" != "0" ]; then
  echo "FATAL: public.schema_migrations still carries the retired migration_name" >&2
  echo "       column after the corpus bootstrap, so the convergence above did not" >&2
  echo "       run and the corpus's own idempotent create no-opped against it." >&2
  echo "       That is the OMN-18544 defect, unconverged. Refusing rather than" >&2
  echo "       resuming against a ledger the corpus cannot write to." >&2
  exit 1
fi

# The bookkeeping column is whatever the corpus made the primary key -- read,
# not written down. Fail closed on anything but exactly one column: a composite
# or absent key means the corpus changed something this loop's ON CONFLICT
# cannot express, and guessing would resume-skip migrations never applied.
KEY_COLUMN="$(psql_db -tAc "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey) WHERE i.indrelid = to_regclass('public.schema_migrations') AND i.indisprimary")"
if [ "$(printf '%s\n' "$KEY_COLUMN" | grep -c .)" != "1" ]; then
  echo "FATAL: public.schema_migrations has no single-column primary key (got:" >&2
  echo "       '${KEY_COLUMN}'). This runner derives its bookkeeping column from" >&2
  echo "       that key and will not guess one (OMN-18544)." >&2
  exit 1
fi
echo "   tracker key column, read from the corpus's own primary key: ${KEY_COLUMN}"

# Carry the retired ledger's rows forward and retire it, in ONE statement so a
# crash cannot strand them. The columns carried are the INTERSECTION of the two
# tables' own catalog entries, so this names only `migration_name` -- the column
# this runner itself created and is retiring -- and nothing about the canonical
# shape. An earlier revision spelled `applied_at` in the INSERT target list,
# which was a literal claim about the corpus's table and exactly the second
# literal this whole change exists to remove.
#
# The drop is conditional on the carry having worked. Dropping unconditionally
# under a message that says "preserving its rows" destroys the only evidence in
# the same statement that would have shown it did not.
psql_db -c "DO \$\$
DECLARE
  stash  regclass := to_regclass('public.schema_migrations_legacy_omn18544');
  live   regclass := to_regclass('public.schema_migrations');
  key_column text;
  carried    text;
  stranded   bigint;
BEGIN
  IF stash IS NULL THEN
    RETURN;
  END IF;
  IF live IS NULL THEN
    RAISE EXCEPTION 'OMN-18544: the retired ledger is stashed but public.schema_migrations does not exist';
  END IF;

  -- STRICT: a composite or absent primary key raises here instead of silently
  -- taking whichever row came first. The shell guard below already refuses that
  -- case, but it runs in a different psql session, so this block carries no
  -- guarantee of its own without the keyword.
  SELECT a.attname INTO STRICT key_column
    FROM pg_index i
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
   WHERE i.indrelid = live AND i.indisprimary;

  SELECT string_agg(quote_ident(s.attname), ', ' ORDER BY s.attnum)
    INTO carried
    FROM pg_attribute s
   WHERE s.attrelid = stash AND s.attnum > 0 AND NOT s.attisdropped
     AND s.attname <> 'migration_name'
     AND s.attname <> key_column
     AND EXISTS (SELECT 1 FROM pg_attribute l
                  WHERE l.attrelid = live AND l.attnum > 0 AND NOT l.attisdropped
                    AND l.attname = s.attname);

  EXECUTE format(
    'INSERT INTO public.schema_migrations (%I%s) SELECT migration_name%s FROM public.schema_migrations_legacy_omn18544 ON CONFLICT DO NOTHING',
    key_column,
    coalesce(', ' || carried, ''),
    coalesce(', ' || carried, ''));

  -- Not a row count: ON CONFLICT DO NOTHING legitimately skips rows a previous
  -- interrupted pass already carried, so a count comparison would refuse its own
  -- re-entrancy. The invariant that holds on every pass is that no stashed key
  -- is left unrepresented.
  EXECUTE format(
    'SELECT count(*) FROM public.schema_migrations_legacy_omn18544 s
       WHERE NOT EXISTS (SELECT 1 FROM public.schema_migrations l WHERE l.%I = s.migration_name)',
    key_column) INTO stranded;
  IF stranded > 0 THEN
    RAISE EXCEPTION 'OMN-18544: % retired ledger row(s) did not carry forward; refusing to drop the stash', stranded;
  END IF;

  RAISE NOTICE 'OMN-18544: retired ledger carried forward into %, stash dropped', key_column;
  DROP TABLE public.schema_migrations_legacy_omn18544;
END
\$\$;"

applied=0
skipped=0
already=0
while IFS="$(printf '\t')" read -r name conditions; do
  [ -n "$name" ] || continue

  if [ "$(psql_db -tAc "SELECT count(*) FROM public.schema_migrations WHERE \"${KEY_COLUMN}\" = '${name}'")" != "0" ]; then
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
  psql_db -c "INSERT INTO public.schema_migrations (\"${KEY_COLUMN}\") VALUES ('${name}')
              ON CONFLICT (\"${KEY_COLUMN}\") DO NOTHING"
  applied=$((applied + 1))
done <<EOF
$(manifest_entries "$MANIFEST")
EOF

echo "== omninode_cloud migration complete: applied=${applied} already=${already} skipped=${skipped} =="
