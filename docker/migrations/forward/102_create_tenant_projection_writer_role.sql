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
-- 1. Role existence. The pg_roles pre-check is what makes this file runnable by
--    a role WITHOUT CREATEROLE: Postgres checks create-role privilege BEFORE it
--    checks whether the name is already taken, so an unconditional CREATE ROLE
--    raises `permission denied to create role` (42501) rather than the
--    duplicate_object the handler below is written for. Same guard shape as
--    094 (app_dashboard), 096 (role_omnidash) and 099 (omninode_runtime).
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tenant_projection_writer') THEN
    BEGIN
      CREATE ROLE tenant_projection_writer WITH
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
-- 2. Fail loud if the role is still not there — never record this migration
--    against a role that does not exist.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tenant_projection_writer') THEN
    RAISE EXCEPTION
      'tenant_projection_writer role does not exist and could not be created — '
      'the executing role lacks CREATEROLE. On a managed instance the role is '
      'provisioned at the provisioning seam (OMN-15343); this migration refuses '
      'to record itself against a role that is not there.';
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 3. CREATEDB / CREATEROLE. Not gated by the executing role's own attributes the
--    way the three below are, but ALTER ROLE still demands role-administration
--    rights, so this is gated on an observed divergence too. Immediately after
--    the guarded CREATE, and on every re-run against a correct role, both flags
--    are already false and no statement is issued at all.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  current_flags RECORD;
BEGIN
  SELECT rolcreatedb, rolcreaterole
    INTO current_flags
    FROM pg_roles
   WHERE rolname = 'tenant_projection_writer';

  IF current_flags.rolcreatedb OR current_flags.rolcreaterole THEN
    BEGIN
      EXECUTE 'ALTER ROLE tenant_projection_writer NOCREATEDB NOCREATEROLE';
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION
          'tenant_projection_writer carries an unexpected privilege '
          '(rolcreatedb=%, rolcreaterole=%) and the executing role lacks the '
          'role-administration rights to correct it — fix this role at the '
          'provisioning seam before the tenant projection path can be trusted',
          current_flags.rolcreatedb, current_flags.rolcreaterole;
    END;
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 4. SUPERUSER / BYPASSRLS / REPLICATION. Privilege-gated by Postgres core: the
--    executing role must already hold an attribute to ALTER it, even to
--    re-assert an already-correct `false`. Only attempt the ALTER when one of
--    the three is actually set — the common case (right after the guarded
--    CREATE, and on every idempotent re-run) never reaches the statement at all,
--    so it never trips the permission error a blanket ALTER produces on RDS.
--
--    This is the load-bearing assertion for the whole P5 cut: rolsuper or
--    rolbypassrls on this role silently exempts every tenant projection write
--    from tenant_isolation RLS, which would turn the isolation proof into a
--    false clean rather than evidence.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  current_flags RECORD;
BEGIN
  SELECT rolsuper, rolbypassrls, rolreplication
    INTO current_flags
    FROM pg_roles
   WHERE rolname = 'tenant_projection_writer';

  IF current_flags.rolsuper
     OR current_flags.rolbypassrls
     OR current_flags.rolreplication THEN
    BEGIN
      EXECUTE 'ALTER ROLE tenant_projection_writer NOSUPERUSER NOBYPASSRLS NOREPLICATION';
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION
          'tenant_projection_writer has an escalated flag (rolsuper=%, '
          'rolbypassrls=%, rolreplication=%) and the executing role lacks '
          'privilege to correct it — a true Postgres superuser must fix this '
          'role manually before tenant projection writes can be trusted to be '
          'subject to RLS',
          current_flags.rolsuper, current_flags.rolbypassrls,
          current_flags.rolreplication;
    END;
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 5. CONNECT on the target database. GRANT ... ON DATABASE is cluster-wide and
--    can be issued from any database context, so this needs no `\connect` and
--    keeps the file inside the flat corpus's one-database rule (099 step 3 uses
--    the identical shape). Guarded on the database existing so the file stays
--    valid on a cluster that has not been through 000.
--
--    This is the topology's declared `object_type: DATABASE, privileges:
--    [CONNECT]` grant for this principal — see
--    `src/omnibase_infra/topology/instances/*.yaml`
--    `principals.tenant_projection_writer.grants`.
-- -----------------------------------------------------------------------------
--    Divergence from 099's step 3: the grant is READ BACK. A GRANT issued by a
--    role that holds neither the database's ownership nor the privilege WITH
--    GRANT OPTION does not raise — PostgreSQL emits `WARNING: no privileges
--    were granted` and returns success. Proven in the scratch replay
--    (2026-08-29, postgres:16-alpine, ordinary NOCREATEROLE login role): 102
--    completed with exit 0 and that warning. Left unchecked this file would
--    record itself as applied on a lane where the principal cannot connect at
--    all, which is the same silent-success class OMN-15819 unmasked for the
--    ledger. The read-back turns it into a named failure.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnidash_analytics') THEN
    EXECUTE 'GRANT CONNECT ON DATABASE omnidash_analytics TO tenant_projection_writer';
    IF NOT has_database_privilege('tenant_projection_writer', 'omnidash_analytics', 'CONNECT') THEN
      RAISE EXCEPTION
        'tenant_projection_writer still lacks CONNECT on omnidash_analytics '
        'after the grant — the executing role (%) can neither grant it nor '
        'inherit it. PostgreSQL only WARNs on a privilege-less GRANT, so this '
        'read-back is the only thing standing between a no-op and a migration '
        'recorded as applied on a lane where the tenant projections cannot '
        'connect. Grant CONNECT at the provisioning seam and re-run.',
        current_user;
    END IF;
  END IF;
END;
$$;
