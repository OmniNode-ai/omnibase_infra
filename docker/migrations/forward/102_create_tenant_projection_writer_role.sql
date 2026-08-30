-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- =============================================================================
-- MIGRATION 102: Create the tenant_projection_writer role (OMN-15425)
-- =============================================================================
-- Ticket: OMN-15425 (P5 — cut tenant projections to the tenant_projection_writer
--         identity). Sibling that already landed: OMN-15426 / OMN-16843
--         (internal projections -> omninode_runtime).
-- Version: 1.0.0
--
-- =============================================================================
-- WHY THIS FILE EXISTS — ENFORCEMENT SHIPPED AHEAD OF PROVISIONING
-- =============================================================================
-- `src/omnibase_infra/topology/application_database.py` already hard-requires
-- this principal for the `tenant_projection` binding:
--
--     _EXPECTED_BINDING_PRINCIPALS = {
--         ...
--         "tenant_projection": "tenant_projection_writer",
--         ...
--     }
--
-- and OMN-16911's `ProjectionBindingConnections.get()` attests
-- `current_user`/`current_database()` on every projection connection against
-- that declaration. The role it demands has never been created by anything.
--
-- Live readback, .201 dev lane (`omnibase-infra-postgres`, database
-- `omnidash_analytics`), 2026-08-29, `select rolname from pg_roles where
-- rolname not like 'pg_%'` — nine roles, none of them this one:
--
--     app_dashboard, omn15683_rls_reader, omn15919_rls_writer,
--     omn16930_rls_reader, omninode_runtime, omninodeadmin, postgres,
--     role_omnidash, role_omniweb
--
-- Consequence, measured the same day on the same lane:
-- `node_projection_delegation_inference_response` DLQ'd 143/143 messages from
-- `onex.evt.omnibase-infra.inference-response.v1` with
--
--     PermissionError: Projection binding 'tenant_projection' connected as
--       ('role_omnidash', 'omnidash_analytics'),
--       expected ('tenant_projection_writer', 'omnidash_analytics')
--
-- The database matched; the login role did not. 100% loss, not one row ever
-- projected on that lane.
--
-- =============================================================================
-- WHAT THIS FILE DOES, AND WHAT IT DELIBERATELY DOES NOT
-- =============================================================================
-- DOES: guard-create the cluster-wide role with the RLS-relevant attributes
-- pinned off, correct an observed attribute divergence when it can, fail loud
-- when it cannot, and grant CONNECT on `omnidash_analytics`.
--
-- DOES NOT carry any credential material. LOGIN + password is a
-- deployment-owned attach by REFERENCE, exactly as OMN-16843 established for
-- `omninode_runtime`: `docker/migrations/forward/000_create_multiple_databases.sh`
-- mints it from `TENANT_PROJECTION_WRITER_PASSWORD` via LOGIN_ONLY_ROLE_MAP
-- (which issues NO grants and NO revokes), and `docker-compose.infra.yml`
-- renders `ONEX_TENANT_DB_URL` from the same variable with the fail-closed
-- `${VAR:?}` form. No password value appears in this file, in the topology, or
-- in any committed compose default.
--
-- DOES NOT grant USAGE on a schema or any table privilege. Those are per-
-- database and this file has no `\connect`: a NEW flat migration whose
-- `\connect` names a database other than `omnibase_infra` is a hard reject
-- under `tests/ci/test_flat_migration_no_foreign_connect_gate.py` /
-- `docker/migrations/forward/cross-database-flat-migrations.yaml`, because the
-- k8s Job that applies this corpus (`omninode_infra`,
-- `k8s/migrations/omnibase-infra-migrate.yaml`) gates its flat `psql -f` loop
-- on `directive_db == "$DB_NAME"` and can never deliver such a file. The
-- schema/table half therefore rides the node-owned loop, which connects
-- directly to `omnidash_analytics` (`NODE_POSTGRES_DB`) — see
-- `docker/migrations/forward/nodes/node_projection_delegation_inference_response/
-- 0004_grant_tenant_projection_writer.sql`. This file is role-only and so is
-- valid in any database context, exactly like 094 (app_dashboard).
--
-- DOES NOT re-assert NOLOGIN on a pre-existing role. 094's reasoning applies
-- unchanged: re-asserting it would REVOKE the deployment-owned credential
-- attach as a side effect of recording a migration. LOGIN is not the isolation
-- control here — NOSUPERUSER, NOBYPASSRLS and non-ownership are, and all three
-- are enforced on every run.
--
-- =============================================================================
-- DESIGN INVARIANTS
-- =============================================================================
--   * NOSUPERUSER + NOBYPASSRLS are ENFORCED on every run, not merely requested
--     at create time. tenant_projection_writer is the identity the OMN-14894
--     tenant_isolation RLS policies are enforced against; a pre-existing role
--     carrying either flag would silently exempt every tenant projection write
--     from the isolation this whole P5 cut exists to establish.
--   * tenant_projection_writer must NEVER own a relation. Postgres exempts a
--     table's owner from RLS unconditionally, FORCE included. Table creation
--     stays with the migration role.
--   * EVERY privileged statement is gated on an OBSERVED DIVERGENCE. ALTER ROLE
--     demands role-administration rights, and SUPERUSER/BYPASSRLS/REPLICATION
--     are additionally gated by the executing role's own attributes — even to
--     re-assert an already-correct `false`. An unconditional ALTER is not
--     "idempotent", it is a privilege demand made on every apply, and it is what
--     makes a file undeliverable under the ordinary, non-CREATEROLE service role
--     the managed lane's migration Job runs as (OMN-15343: that instance has no
--     `postgres` role at all). This file reads pg_roles first and issues a
--     statement only when observed state differs from required state.
--   * When a divergence IS observed and the executing role cannot correct it,
--     raise an explicit exception naming the flag. Never silently succeed, never
--     silently leave an escalation in place.
--
-- IDEMPOTENCY:
--   Safe to re-run. The CREATE is skipped when pg_roles already shows the role
--   and still carries the duplicate_object/unique_violation guard for the
--   genuine race (roles are cluster-wide; two migration paths may race). Every
--   ALTER runs only on an observed divergence, so a correct role touches none of
--   them. The CONNECT grant is idempotent in Postgres by construction.
--
-- ROLLBACK:
--   See rollback/rollback_102_create_tenant_projection_writer_role.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Role existence and attributes. These use psql \gexec instead of a DO block
--    so the application SQL gate can statically inspect the file. Each command
--    is emitted only when catalog state proves it is needed.
-- -----------------------------------------------------------------------------
SELECT CASE
  WHEN NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'tenant_projection_writer')
    THEN 'true'
  ELSE 'false'
END AS create_tenant_projection_writer \gset

\if :create_tenant_projection_writer
CREATE ROLE tenant_projection_writer WITH NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
\endif

SELECT 'ALTER ROLE tenant_projection_writer NOCREATEDB NOCREATEROLE'
WHERE EXISTS (
  SELECT 1
    FROM pg_catalog.pg_roles
   WHERE rolname = 'tenant_projection_writer'
     AND (rolcreatedb OR rolcreaterole)
) \gexec

SELECT 'ALTER ROLE tenant_projection_writer NOSUPERUSER NOBYPASSRLS NOREPLICATION'
WHERE EXISTS (
  SELECT 1
    FROM pg_catalog.pg_roles
   WHERE rolname = 'tenant_projection_writer'
     AND (rolsuper OR rolbypassrls OR rolreplication)
) \gexec

-- Fail if the role still does not exist.
SELECT 'tenant_projection_writer'::regrole;

-- -----------------------------------------------------------------------------
-- 2. CONNECT on the target database. Guarded on the database existing so the
--    file stays valid on a cluster that has not been through 000.
-- -----------------------------------------------------------------------------
SELECT 'GRANT CONNECT ON DATABASE omnidash_analytics TO tenant_projection_writer'
WHERE EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omnidash_analytics') \gexec

SELECT CASE
  WHEN NOT EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omnidash_analytics')
    THEN 1
  WHEN has_database_privilege('tenant_projection_writer', 'omnidash_analytics', 'CONNECT')
    THEN 1
  ELSE 1 / (
    SELECT count(*)
      FROM pg_catalog.pg_database
     WHERE datname = 'omnidash_analytics'
       AND has_database_privilege('tenant_projection_writer', 'omnidash_analytics', 'CONNECT')
  )
END;
