-- onex-create-database: omnidash_analytics
-- =============================================================================
-- TOMBSTONE (OMN-15819, 2026-08-10): this file is UNDELIVERABLE via the k8s
-- Job that applies docker/migrations/forward/*.sql
-- (omninode_infra/k8s/migrations/omnibase-infra-migrate.yaml). That Job owns
-- only the omnibase_infra database; its flat loop's `psql -f` apply is
-- gated on `directive_db == $DB_NAME` and is UNREACHABLE for this file's
-- `\connect omnidash_analytics` below, in that loop or any other in the
-- runner. It has never executed anywhere -- live-confirmed 2026-08-10
-- (role_omnidash, omninode-dev-postgres RDS): `to_regclass('omninode_internal
-- .live_events')` fails closed with `permission denied for schema
-- omninode_internal`, and the runtime write path has been failing every
-- write with UndefinedTable since before this file merged. Kept in place,
-- byte-unchanged below this header, as ledgered history (migration files are
-- append-only) -- do NOT delete it and do NOT try to make it deliverable in
-- place. The table-creation + omninode_runtime-grant half of this file's
-- intent is DELIVERED by the node-owned replacement migration,
-- docker/migrations/forward/nodes/node_projection_live_events/0002_create_omninode_internal_live_events.sql
-- (connects directly to omnidash_analytics as role_omnidash -- the one code
-- path in the runner that can actually reach this database). That
-- replacement does NOT duplicate this file's transform-copy/reconciliation
-- of pre-existing public.live_events rows -- see its own header for scope.
-- Runner honesty for this class of file (no silent false-"applied" ledger
-- row; fail-closed for any FUTURE cross-DB flat file) is the companion PR in
-- omninode_infra. Static pre-merge enforcement in THIS repo:
-- tests/ci/test_flat_migration_no_foreign_connect_gate.py /
-- docker/migrations/forward/cross-database-flat-migrations.yaml.
-- =============================================================================
-- AMENDMENT (OMN-15838, 2026-08-10): the "byte-unchanged below this header"
-- promise above is superseded for the transform-copy/reconciliation section
-- ONLY. The tombstone above is accurate for its own scope (the k8s Job /
-- managed-RDS path this file cannot reach) but that is not the only path
-- that executes this file: `scripts/run-forward-migrations.sh` (the .201
-- docker-compose lanes -- dev/stability-test/prod) applies this exact flat
-- file directly via `psql -v ON_ERROR_STOP=1 -f`, tracked one-time per
-- database in that lane's own `public.schema_migrations` (filename-keyed,
-- content-blind -- confirmed by reading the runner script, OMN-15838). On
-- that path this file DOES execute, and its transform-copy + exact-row-count
-- reconciliation (formerly step 2/3 below) is a non-atomic race: it snapshots
-- `public.live_events` via one INSERT...SELECT, then re-counts both tables in
-- separate follow-up statements. Under READ COMMITTED, any row committed to
-- `public.live_events` by the live write path between the INSERT and the
-- recount (observed at ~24 writes/min on the stability lane) makes
-- `v_dst_count <> v_src_count` true by construction, RAISE EXCEPTIONs, and
-- rolls back the INSERT within that DO block -- deterministically failing on
-- any lane with concurrent writers, not intermittently. Because
-- `ON_ERROR_STOP=1` aborts the whole runner script on that RAISE, the
-- migration is never recorded applied and every subsequent refresh attempt
-- retries and fails identically (OMN-15838: 963965 vs 963977 divergence,
-- blocking the OMN-15826/OMN-15837 stability-test refresh outright).
--
-- Since data delivery for this table is already owned by the node-owned
-- replacement migration this file's own tombstone names above
-- (0002_create_omninode_internal_live_events.sql, which does NOT duplicate
-- the transform-copy), 099's own copy of that logic is now dead weight on
-- top of being racy. This amendment REMOVES the transform-copy +
-- reconciliation DO block entirely; 099 is now schema-shape + grant only
-- (CREATE TABLE, GRANT, ALTER DEFAULT PRIVILEGES, indexes, and the two
-- statically-provable post-condition assertions it already carried) on every
-- path that executes it. See tests/integration/migrations/
-- test_099_omninode_internal_live_events_omn15359.py for the updated proof
-- set (row-copy/reconciliation tests replaced by a direct regression test
-- that a diverging public/internal row count no longer fails the migration).
--
-- RESIDUAL NOT FIXED HERE (OMN-15838, noted for the refresh to absorb): the
-- stability-test lane's `omninode_internal.live_events` currently carries
-- ~108MB of index pages built during an earlier successful apply (when the
-- table held ~963965 rows) over what is now an EMPTY table -- the failed
-- re-run's RAISE rolled back that attempt's INSERT (rolling back everything
-- executed inside that DO block) but ran strictly before the CREATE INDEX
-- statements below it in file order, so the indexes from the earlier
-- successful apply were never touched and never shrink on their own
-- (ordinary Postgres MVCC/index bloat, not a bug in the index statements
-- themselves). A REINDEX is a storage-maintenance operation, not a
-- schema-shape assertion, and this migration only ever runs once per
-- database (ledger-gated); adding it here would need to be correct for both
-- the fresh-create and the bloated-reapply case for a benefit that only
-- matters once. Left as an out-of-band step for whichever refresh/recovery
-- procedure next lands the fixed 099 on that lane: `REINDEX TABLE
-- omninode_internal.live_events;` (or DROP + recreate the 4 indexes below)
-- after this migration reports success.
-- =============================================================================
-- MIGRATION: physically create omninode_internal.live_events (schema shape +
--            grants only -- OMN-15359 original, OMN-15838 removed the
--            transform-copy/reconciliation of public.live_events rows)
-- =============================================================================
-- Ticket: OMN-15359 (P2-P4 build classified schemas and migrate internal,
--         control-plane, catalog, and tenant targets)
-- Related: OMN-15421 (Projection Domain Adapter Proof), OMN-15423 (relation
--          inventory/ownership), OMN-15425 (tenant projection authority --
--          separate, unrelated gap), OMN-15819 (node-owned replacement now
--          owns data delivery), OMN-15838 (this amendment -- removed the
--          racy transform-copy/reconciliation block)
-- Version: 1.1.0
--
-- WHAT THIS FILE DOES
--   098_create_omninode_internal_schema.sql created the (empty) omninode_internal
--   schema and a temporary physical bridge (physical_schema_mapping.py's
--   INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359) that let
--   omninode_internal-domain tables keep resolving against `public` where they
--   physically still live. This migration performs the first real family
--   cutover out of that bridge: `live_events`.
--
--   1. CREATE TABLE omninode_internal.live_events, identical shape to
--      public.live_events (docker/migrations/forward/nodes/node_projection_live_events/
--      0000_create_live_events.sql + 0001's type-classification data fixups).
--   2. public.live_events is PRESERVED -- not dropped, not ALTER TABLE ... SET
--      SCHEMA'd, and (as of OMN-15838) not read from at all by this file.
--      Backfilling `omninode_internal.live_events` from pre-existing
--      `public.live_events` history is intentionally out of scope here -- see
--      the node-owned replacement migration's own header for the same
--      framing ("a distinct, separately-scoped follow-up if the dashboard
--      needs continuous history in the internal-schema copy").
--
-- WHY THIS TABLE, WHY NOW
--   node_projection_live_events/contract.yaml has declared
--   db_io.db_tables[0].schema: omninode_internal since before this migration
--   existed. handler_wiring.py's _resolve_projection_database_target uses that
--   contract-declared schema literally as the SQL write target (it does not
--   consult the physical bridge for query building, only for the grant-
--   privilege check) -- so the runtime has been issuing
--   `INSERT INTO omninode_internal.live_events` since before 098 merged, and
--   failing with "relation does not exist" because no migration had ever
--   physically created that relation. 098 fixed the schema-level half of that
--   gap; this migration fixes the table-level half for this one family.
--
-- WHY live_events IS ALSO REMOVED FROM THE BRIDGE (companion change, same PR)
--   src/omnibase_infra/topology/physical_schema_mapping.py drops "live_events"
--   from INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359 in this same PR.
--   Post-migration, physical_grant_schema_for_table('omninode_internal',
--   'live_events') must agree with the INSERT-target schema
--   (omninode_internal) -- leaving it in the bridge after the physical table
--   exists would make the grant-privilege check assert against `public` while
--   every write already lands in `omninode_internal`, silently reintroducing
--   the same class of drift this migration exists to close. The shipped
--   topology TABLE grants (src/omnibase_infra/topology/instances/*.yaml,
--   docker/catalog/database-topology/*.yaml) are regenerated in the same PR
--   via scripts/generate_application_database_table_grants.py --write so the
--   omninode_runtime principal's live_events grant moves from schema: public
--   to schema: omninode_internal, matching physical reality.
--
-- DATABASE CONTEXT
--   Same pattern as 083/096/097/098: the forward runner applies this against
--   POSTGRES_DB, so this file switches with `\connect omnidash_analytics`.
--
-- IDEMPOTENCY
--   CREATE TABLE IF NOT EXISTS, the GRANT statements, and CREATE INDEX IF NOT
--   EXISTS are all safe to re-run.
--
-- ROLLBACK
--   See rollback/rollback_099_create_omninode_internal_live_events.sql.
--   RESTRICT, not CASCADE, and refuses (by construction -- DROP TABLE with no
--   CASCADE on a table nothing else yet depends on) once any downstream
--   object has been built against omninode_internal.live_events. Because
--   public.live_events is untouched, rollback of this migration alone never
--   loses data: the source of truth remains intact throughout.
--
-- WHY THIS FILE ALSO ISSUES THE omninode_runtime GRANT (grant-gap repair,
-- same PR, projplane-slice-verify)
--   098 physically created the omninode_internal SCHEMA but deliberately
--   granted nothing (its own docstring lists "owner_omninode_internal/
--   omninode_runtime role creation" as explicit deferred P5 work). This file
--   physically creates the FIRST TABLE in that schema and simultaneously
--   removes live_events from INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
--   -- so, unlike the other ~40 still-bridged families, live_events is live
--   TODAY: handler_wiring._resolve_projection_database_target already issues
--   `INSERT INTO omninode_internal.live_events` against the omninode_runtime
--   binding in production. A migration that creates the table but never
--   grants the write-path role converts the prior UndefinedTable failure into
--   InsufficientPrivilege on first deploy -- proven live against a real RDS
--   snapshot (4 independent checks: the table is owned by a role
--   omninode_runtime is not; `information_schema.role_table_grants` returns
--   zero rows for omninode_runtime against this table; omninode_runtime has
--   no CREATE on omninode_internal; `pg_default_acl` carries no entry for the
--   schema). The GRANT below is derived from, and must stay byte-for-byte
--   consistent with, the single declared source of truth:
--   `src/omnibase_infra/topology/instances/*.yaml` `principals.omninode_runtime
--   .grants[schema: omninode_internal]` (regenerated into
--   `docker/catalog/database-topology/*.yaml` by
--   `scripts/generate_application_database_table_grants.py --write`) --
--   SELECT, INSERT, UPDATE, no DELETE (a projection writer upserts, it does
--   not reshape the table -- same invariant 096 states for role_omnidash).
--   No other principal declares a live_events grant in that topology today,
--   so no read-side principal needs anything here.
--
-- WHY THE ROLE IS ALSO GUARD-CREATED HERE (not just granted)
--   Unlike 098's deferred table-privilege work, `omninode_runtime` itself has
--   never been created by ANY migration in this repo (grepped the full
--   `docker/migrations/forward/*.sql` corpus: zero `CREATE ROLE omninode_runtime`
--   hits before this file). The standalone "Migration Integration Test" CI
--   gate applies only `docker/migrations/forward/*.sql` via
--   `scripts/run-migrations.py` against a bare `postgres:16-alpine` service --
--   it never runs `000_create_multiple_databases.sh` (that script's own
--   SERVICE_DB_MAP does not even list `omninode_runtime`; it is a distinct,
--   newer per-domain principal, not one of the legacy per-microservice
--   `role_*` names). Without a guarded CREATE here, the GRANT below would
--   fail closed with `role "omninode_runtime" does not exist` (42704) in
--   that CI scope on every PR, not just this one. The guard follows the
--   exact precedent 094 (app_dashboard) and 096 (role_omnidash) already
--   established in this file set: CREATE only when absent (so a role
--   provisioned out-of-band on a managed instance is never re-created or
--   clobbered), NOLOGIN at create time (LOGIN + password attach stays
--   deployment-owned -- AWS Secrets Manager on the cloud path, lane-local on
--   compose -- never re-asserted here even though the topology declares
--   `login: true`), and RAISE EXCEPTION rather than silently skip if the
--   executing role lacks CREATEROLE and the role is not already there.
--
-- WHY ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal (decision, not a
-- given)
--   098's own docstring names ~40 more OMNINODE_INTERNAL-domain families
--   still physically bridged to `public`, each of which will need this exact
--   same GRANT boilerplate the day its own family-cutover migration lands.
--   Repeating steps 1-4 by hand in each of those ~40 files is exactly the
--   defect class this repair exists to close -- a future author who copies
--   099's CREATE TABLE but forgets the GRANT reintroduces the identical
--   UndefinedTable->InsufficientPrivilege bug this file fixes. `ALTER DEFAULT
--   PRIVILEGES IN SCHEMA omninode_internal GRANT ... TO omninode_runtime`
--   makes every future table created BY THE SAME EXECUTING ROLE in this
--   schema inherit the grant automatically, with no per-migration GRANT
--   statement required. The scope is correct because it is bounded to the
--   caveat 096 already documents for `public`: default privileges apply only
--   to objects the CURRENT (migration-executing) role creates -- exactly the
--   role every future omninode_internal family-cutover migration will run
--   as, on every lane (compose: `postgres`; managed: the OMN-15335 migration
--   principal). If a lane's migrations ever ran as a different principal than
--   the one that issued this ALTER DEFAULT PRIVILEGES, that principal would
--   need its own default-privileges assertion -- the same fact 096 already
--   states for `public` and not a new caveat introduced here.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Role existence (cluster-wide, issued before \connect). Guarded exactly
--    like 094/096: Postgres checks create-role privilege BEFORE it checks
--    whether the name is taken, so an unconditional CREATE ROLE would raise
--    `permission denied to create role` (42501) instead of the
--    duplicate_object this handler is written for.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omninode_runtime') THEN
    BEGIN
      CREATE ROLE omninode_runtime WITH
        NOLOGIN
        NOSUPERUSER
        NOBYPASSRLS
        NOCREATEDB
        NOCREATEROLE
        NOREPLICATION;
    EXCEPTION
      WHEN duplicate_object OR unique_violation THEN
        NULL; -- created concurrently by another migration path
    END;
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 2. Fail loud, name the problem, if the role could not be created and is
--    still not there -- never silently record this migration against a role
--    that does not exist.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omninode_runtime') THEN
    RAISE EXCEPTION
      'omninode_runtime role does not exist and could not be created -- the '
      'executing role lacks CREATEROLE. On a managed instance the role is '
      'provisioned at the provisioning seam; this migration refuses to record '
      'itself against a role that is not there.';
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 3. CONNECT on the target database, issued from the current (omnibase_infra)
--    context because GRANT ... ON DATABASE is cluster-wide. Guarded on the
--    database existing so this file stays valid on a cluster that has not
--    been through 000 -- mirrors 096 step 3 exactly.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnidash_analytics') THEN
    EXECUTE 'GRANT CONNECT ON DATABASE omnidash_analytics TO omninode_runtime';
  END IF;
END;
$$;

\connect omnidash_analytics

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS omninode_internal.live_events (
  id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id       TEXT        UNIQUE NOT NULL,
  type           TEXT        NOT NULL DEFAULT 'ACTION',
  timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  source         TEXT        NOT NULL DEFAULT 'platform',
  topic          TEXT        NOT NULL DEFAULT '',
  summary        TEXT        NOT NULL DEFAULT '',
  payload        TEXT        NOT NULL DEFAULT '{}',
  correlation_id TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- 4. The actual grant-gap repair. Derived from, and must stay consistent
--    with, `src/omnibase_infra/topology/instances/*.yaml`
--    `principals.omninode_runtime.grants[schema: omninode_internal]`.
--    SELECT/INSERT/UPDATE only -- no DELETE, matching the topology
--    declaration and 096's projection-writer invariant.
-- -----------------------------------------------------------------------------
GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.live_events TO omninode_runtime;

-- -----------------------------------------------------------------------------
-- 5. Forward-looking default privileges for the ~40 remaining
--    OMNINODE_INTERNAL-domain families still bridged to `public` (098's
--    deferred-work list). See the file-header rationale above for scope and
--    the "same executing role" caveat this inherits from 096.
-- -----------------------------------------------------------------------------
ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal
  GRANT SELECT, INSERT, UPDATE ON TABLES TO omninode_runtime;

-- No transform-copy / reconciliation block here (OMN-15838 removed it -- see
-- the AMENDMENT in the file header for why: it was a non-atomic race against
-- a live `public.live_events` writer, and data delivery for this table is
-- already owned by the node-owned replacement migration, not this file).
-- omninode_internal.live_events is therefore created empty by this file on
-- every path that executes it -- schema shape and grants only.

-- Indexes matching public.live_events's shape
-- (docker/migrations/forward/nodes/node_projection_live_events/0000_create_live_events.sql).
CREATE INDEX IF NOT EXISTS idx_omninode_internal_live_events_created_at
  ON omninode_internal.live_events (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_omninode_internal_live_events_topic
  ON omninode_internal.live_events (topic);

CREATE INDEX IF NOT EXISTS idx_omninode_internal_live_events_source
  ON omninode_internal.live_events (source);

CREATE INDEX IF NOT EXISTS idx_omninode_internal_live_events_correlation_id
  ON omninode_internal.live_events (correlation_id)
  WHERE correlation_id IS NOT NULL;

-- Post-condition. Statically provable (no DO/RAISE), matching the
-- OMN-15361 application-database gate's requirement for deployable SQL.
SELECT 1 / count(*) AS omninode_internal_live_events_exists_assertion
  FROM information_schema.tables
 WHERE table_schema = 'omninode_internal' AND table_name = 'live_events';

-- Post-condition on the grant-gap repair itself: the write-path role must
-- actually carry INSERT on the physical table this migration just created,
-- or the migration Job succeeded while leaving the runtime write path
-- broken. Statically provable, same pattern as the assertion above.
SELECT 1 / count(*) AS omninode_runtime_live_events_insert_grant_assertion
  FROM information_schema.role_table_grants
 WHERE table_schema = 'omninode_internal'
   AND table_name = 'live_events'
   AND grantee = 'omninode_runtime'
   AND privilege_type = 'INSERT';
