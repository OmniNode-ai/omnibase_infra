-- onex-create-database: omnidash_analytics
-- =============================================================================
-- TOMBSTONE (OMN-15846, 2026-08-10): this file is UNDELIVERABLE via the k8s
-- Job that applies docker/migrations/forward/*.sql
-- (omninode_infra/k8s/migrations/omnibase-infra-migrate.yaml) and, on
-- onex-dev RDS today, LATENT rather than live-broken besides.
--
-- UNDELIVERABLE: that Job owns only the omnibase_infra database; its flat
-- loop's `psql -f` apply is gated on `directive_db == $DB_NAME` and is
-- UNREACHABLE for this file's `\connect omnidash_analytics` below, in that
-- loop or any other in the runner. It has never executed there --
-- live-confirmed 2026-08-10 (omninode-dev-postgres RDS): app_dashboard's
-- USAGE on schema public is granted BY `pg_database_owner`
-- (`nspacl: app_dashboard=U/pg_database_owner`), never by
-- role_omnibase_infra (the identity this file would have executed under)
-- -- this file's own step 3 `GRANT USAGE ON SCHEMA public TO app_dashboard`
-- left no trace. omnibase_infra's own schema_migrations carried a false
-- "applied" row for this file (applied_at 2026-08-01T22:00:19Z, checksum
-- byte-identical to this file's live content) -- the same OMN-15819
-- masking class OMN-15846's classification-ordering fix unmasked for
-- 098/099.
--
-- LATENT, NOT LIVE-BROKEN: this file's own header (the "THE DEFECT" section
-- below, unchanged) documents that the CONNECT gap it repairs is latent
-- until PUBLIC's CONNECT is revoked platform-wide (OMN-15355) -- "not an
-- edge case being defended against ... the declared target state." Live
-- readback 2026-08-10 confirms PUBLIC's CONNECT on omnidash_analytics has
-- NOT been revoked on this RDS instance (`pg_database.datacl` carries
-- `=Tc/omninodeadmin`, i.e. PUBLIC still holds CONNECT), so app_dashboard
-- connects today via that still-live PUBLIC default, not via this file.
-- OMN-15355 (P1, "In Review" as of 2026-08-10, explicit acceptance
-- criterion naming app_dashboard by name) is the tracked, systematic
-- successor that will retire PUBLIC's default CONNECT under its own
-- generated ACL matrix and change-window gate -- it is the correct owner
-- of app_dashboard's post-revocation CONNECT grant, not a flat migration
-- this k8s Job cannot deliver.
--
-- Kept in place, byte-unchanged below this header, as ledgered history
-- (migration files are append-only) -- do NOT delete it and do NOT try to
-- make it deliverable in place; it remains applicable, unchanged, on any
-- lane whose own runner (scripts/run-forward-migrations.sh, a different
-- code path from the k8s Job this tombstone concerns) can actually reach
-- omnidash_analytics. Static pre-merge enforcement in THIS repo:
-- tests/ci/test_flat_migration_no_foreign_connect_gate.py /
-- docker/migrations/forward/cross-database-flat-migrations.yaml.
-- =============================================================================
-- MIGRATION: app_dashboard CONNECT + policy-gated read grants on omnidash_analytics
-- =============================================================================
-- Ticket: OMN-15297 (the grant chain never grants CONNECT — the read role
--         cannot open a session at all)
-- Related: OMN-14899/094 (the role), OMN-14894/0023 (the RLS policies + table
--          grants), OMN-15363/096 (same repair for the writer role_omnidash),
--          OMN-15355 (revoke PUBLIC's CONNECT — the change that makes this
--          latent defect fatal)
-- Version: 1.0.0
--
-- THE DEFECT
--   094 creates app_dashboard. Node migration 0023 grants USAGE ON SCHEMA
--   public and SELECT on the two RLS-covered delegation tables. Nothing in
--   that chain ever grants CONNECT ON DATABASE. On a stock Postgres CONNECT is
--   held by PUBLIC, so the gap is LATENT and every test passes. On a database
--   where PUBLIC's CONNECT has been revoked the role cannot open a session at
--   all and every grant behind it is unreachable:
--
--     FATAL:  permission denied for database "omnidash_analytics"
--     DETAIL:  User does not have CONNECT privilege.
--
--   Live readback, .201 dev lane 2026-07-28 (compose project
--   `omnibase-infra`, db `omnidash_analytics`), with LOGIN attached:
--
--     rolname       | rolcanlogin | rolsuper | rolbypassrls | can_connect
--     app_dashboard | t           | f        | f            | f
--
--   This reads like a credential problem and is a missing grant. OMN-15355
--   revokes PUBLIC's CONNECT platform-wide by design, so this is not an
--   edge case being defended against — it is the declared target state.
--
-- WHY AN ADDITIVE FILE AND NOT AN EDIT TO 0023
--   0023 is a VENDORED node migration: the authoritative copy lives in
--   omnimarket (`src/omnimarket/nodes/node_projection_delegation/migrations/`)
--   and is mirrored here by `scripts/sync-node-migrations.sh`. Editing the
--   mirror would be silently reverted by the next sync. 0023 is also on the
--   operator fence (`FENCED_NODE_MIGRATION_IDS`, OMN-15336), so a fix placed
--   inside it would not execute on any fenced lane — the repair has to live
--   where it can actually run.
--
-- WHY app_dashboard AND NOT role_omnidash
--   Two roles, two paths, both real: role_omnidash is the projection WRITER
--   (096) and app_dashboard is the dashboard READ role (094). This file is the
--   read half only. It grants no DML anywhere.
--
-- WHAT THIS FILE IS NOT
--   * It carries NO credential material and does NOT grant LOGIN. The
--     LOGIN + password attach is deployment-owned and operator-gated (AWS
--     Secrets Manager, `omninode/staging/rds/app-dashboard`), exactly as 094
--     states. Re-asserting NOLOGIN would REVOKE that attach.
--   * It does NOT revoke PUBLIC's CONNECT. That is OMN-15355, it has real
--     blast radius across every role and database on the instance, and it is
--     not this ticket's to take unilaterally.
--   * It does NOT grant CREATE ON DATABASE or CREATE ON SCHEMA. A role that
--     can create objects OWNS them, and an owner is exempt from row-level
--     security — FORCE included. Granting CREATE to the read role would
--     reopen, for every future table, precisely the bypass OMN-14894 exists
--     to close.
--   * It does NOT grant SELECT on the projection VIEWS (0010). Postgres
--     evaluates RLS against the VIEW OWNER, so reading through an owner's view
--     silently bypasses tenant isolation until the views are recreated with
--     `security_invoker = true`. Step 3 filters to `relkind = 'r'` for that
--     reason, not by accident.
--
-- TABLE GRANTS STAY WITH 0023
--   The table-level SELECT grants stay in vendored node migration 0023, next
--   to the tenant_isolation policies they depend on. Top-level migration 097
--   is applied before node migrations on fresh asyncpg lanes, so relation
--   grants here would either fail before the tables exist or require dynamic
--   SQL that the OMN-15361 application database gate rightly rejects.
--
-- DATABASE CONTEXT
--   The forward runner applies `docker/migrations/forward/*.sql` against
--   POSTGRES_DB (`omnibase_infra`), so this file switches with
--   `\connect omnidash_analytics` partway through — the established in-repo
--   pattern (083, 096). GRANT ... ON DATABASE is cluster-wide and is issued
--   BEFORE the switch; every schema/table grant is issued after it. The
--   `onex-create-database` directive on line 1 is honoured by
--   `ensure_directive_database` in `scripts/run-forward-migrations.sh`, so the
--   `\connect` cannot fail on a cluster that has not been through
--   `000_create_multiple_databases.sh`.
--
-- IDEMPOTENCY
--   Safe to re-run. GRANT/REVOKE are idempotent by definition, and every
--   statement that is not is gated on an observed catalog read. No ALTER ROLE
--   appears anywhere in this file: an unconditional ALTER is not idempotent,
--   it is a privilege demand made on every apply, and it is what stopped 094
--   applying under the RDS master (OMN-14899) and then under the ordinary
--   service role the k8s Job uses (OMN-15343).
--
-- EXECUTING-ROLE REQUIREMENTS
--   GRANT requires the executing role to hold the privilege WITH GRANT OPTION
--   or to be the object owner. On compose lanes that is `postgres`; on the
--   managed instance it is the per-database migration principal, which owns
--   the database and the projection tables (OMN-15335). No CREATEROLE and no
--   superuser is required by this file.
--
-- ROLLBACK
--   See rollback/rollback_097_grant_app_dashboard_connect_omnidash_analytics.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. The read role must already exist. It is created by 094, which runs first
--    (filename order). The direct GRANT below fails closed if the role is
--    absent, without using a procedural block that OMN-15361 cannot statically
--    prove.
-- -----------------------------------------------------------------------------

-- -----------------------------------------------------------------------------
-- 2. CONNECT on the target database — the defect this ticket names.
--    Issued from the current (omnibase_infra) context because
--    GRANT ... ON DATABASE is cluster-wide, and guarded on the database
--    existing so this file is valid on a cluster that has not been through
--    000_create_multiple_databases.sh.
--
--    The database name is a literal, not current_database(): a plain GRANT has
--    no current_database() shorthand, and the forward runner is connected to
--    omnibase_infra at this point, so resolving it dynamically would grant
--    CONNECT on the WRONG database.
-- -----------------------------------------------------------------------------
GRANT CONNECT ON DATABASE omnidash_analytics TO app_dashboard;

\connect omnidash_analytics

-- -----------------------------------------------------------------------------
-- 3. Schema resolution. USAGE only — CREATE is neither granted (see "WHAT THIS
--    FILE IS NOT") nor revoked (revoking it from the migration principal would
--    break the cloud migration path as a side effect, OMN-15335). USAGE alone
--    confers no data access; every row still goes through step 3b.
-- -----------------------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO app_dashboard;

-- -----------------------------------------------------------------------------
-- 3b. Table-level SELECT is deliberately absent here. Vendored node migration
--     0023 owns those grants because it also owns the tenant_isolation policy
--     creation and runs after the delegation tables exist.
-- -----------------------------------------------------------------------------

-- -----------------------------------------------------------------------------
-- 4. Post-conditions. Grants are not the isolation control — the role flags
--    are — so they are asserted, not assumed.
--
--    These SELECT assertions deliberately avoid DO/RAISE so the deployable SQL
--    remains statically provable by OMN-15361.
-- -----------------------------------------------------------------------------
SELECT 1 / count(*) AS app_dashboard_connect_assertion
  FROM (
    SELECT 1
     WHERE has_database_privilege('app_dashboard', current_database(), 'CONNECT')
  ) AS assertion;

SELECT 1 / count(*) AS app_dashboard_rls_role_flags_assertion
  FROM (
    SELECT 1
      FROM pg_catalog.pg_roles
     WHERE rolname = 'app_dashboard'
       AND NOT rolsuper
       AND NOT rolbypassrls
  ) AS assertion;
