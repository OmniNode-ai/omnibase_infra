-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- =============================================================================
-- MIGRATION 103: Create the tenant_projection_writer role (OMN-15425)
-- =============================================================================
-- Ticket: OMN-15425 (P5 — cut tenant projections to the tenant_projection_writer
--         identity). Sibling that already landed: OMN-15426 / OMN-16843
--         (internal projections -> omninode_runtime).
-- Version: 2.0.0 (OMN-17301: guarded against the privileges the managed
--          lane's migration identities do not hold)
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
--   See rollback/rollback_103_create_tenant_projection_writer_role.sql
-- =============================================================================


-- =============================================================================
-- OMN-17301 — WHY THE PRIVILEGED STATEMENTS ARE GUARDED, NOT MERELY GATED
-- =============================================================================
-- The 1.0.0 revision of this file gated every privileged statement on an
-- OBSERVED DIVERGENCE (read pg_roles first, act only on a difference) and
-- treated that as sufficient to make the file deliverable under an ordinary,
-- non-CREATEROLE service role. It is not, and the gap was structural rather
-- than incidental: a divergence gate answers "should this statement run?", it
-- does not answer "may this role run it?". On the managed lane the answer to
-- the first question was YES (the role was absent, so it had to be created)
-- and to the second it was NO — and nothing in the file bridged that.
--
-- Measured, `Deploy onex-staging` run 33341217605 (job 99337209081, pod
-- omnibase-infra-migrate-s5mzs, 2026-08-30T23:12:58Z):
--
--     APPLY: 103_create_tenant_projection_writer_role.sql
--         execution role: role_omnibase_infra (cluster role DDL)
--     psql:/work/103_create_tenant_projection_writer_role.sql:146: ERROR:
--       permission denied to create role
--     DETAIL:  Only roles with the CREATEROLE attribute may create roles.
--
-- The migrate Job is migration-order 1 of 6 and runs BEFORE overlay-apply and
-- the runtime digest pin, so this aborted EVERY staging deploy on every
-- trigger, not merely the one that surfaced it: the OMN-16493 resolver picks
-- the NEWEST CI-built migrate bundle fail-closed, and every bundle at or after
-- c5a3c2d27 carries this file.
--
-- REPRODUCED, not inferred (OMN-17301, scratch PostgreSQL 16, executing role
-- created LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE): the 1.0.0 bytes fail at
-- the identical statement with the identical SQLSTATE 42501 and the identical
-- file line number 146. See tests/unit/db/test_migration_103_omn17301.py.
--
-- THREE defects were found by that reproduction, not one. All three are fixed
-- below and all three are covered by that module:
--
--   D1  CREATE ROLE aborted the file with a RAW psql permission error. The
--       CREATE was wrapped only in a duplicate_object/unique_violation handler
--       — the race case — with nothing for insufficient_privilege, which is the
--       case the managed lane actually presents on every run. Fixed in §1: the
--       handler now names the condition, the executing role, and the exact
--       provisioning-seam remediation.
--
--   D2  GRANT CONNECT SILENTLY DID NOTHING. `omnidash_analytics` is owned by
--       role_omnidash on the managed lane, not by the flat loop's
--       role_omnibase_infra, so the grantor holds no grant option on it.
--       PostgreSQL does not raise for that — it emits
--       `WARNING: no privileges were granted for "omnidash_analytics"` and
--       returns success. Reproduced on a scratch instance shaped with the real
--       ownership split. Fixed in §2: the outcome is now READ BACK rather than
--       assumed, and the ineffective case is reported explicitly.
--
--   D3  THE CONNECT ASSERTION WAS VACUOUS — the OMN-14950 class. It asserted
--       `has_database_privilege('tenant_projection_writer', ..., 'CONNECT')`,
--       which is TRUE on any database that has not had PUBLIC's default CONNECT
--       revoked, regardless of whether the grant this file issues ever landed.
--       Proven on the same scratch instance: with the explicit grant absent,
--       `datacl` read `=Tc/role_omnidash,role_omnidash=CTc/role_omnidash` — the
--       leading `=Tc` IS the PUBLIC grant — and the assertion still passed. It
--       therefore could not detect D2, which is why D2 survived review. Fixed
--       in §2: the readback distinguishes an explicit role grant from PUBLIC's
--       default, and says which one is carrying the privilege.
--
-- WHY THE ROLE IS NOT CREATED HERE ON THE MANAGED LANE, AND WHY THAT IS NOT
-- A MASKING SKIP
-- ---------------------------------------------------------------------------
-- PostgreSQL roles are CLUSTER-scoped. The migrate Job holds exactly two
-- credentials against the managed instance — role_omnibase_infra (flat loop)
-- and role_omnidash (node loop) — and BOTH are provisioned NOCREATEROLE by
-- contract (omninode_infra scripts/init-databases.sh: "no CREATEDB, no
-- SUPERUSER, no CREATEROLE"). There is no escalation identity in the Job BY
-- DESIGN: the managed instance has no `postgres` role at all, its master
-- credential is held by AWS Secrets Manager under terraform
-- `manage_master_user_password`, and OMN-15335 states the rule directly — an
-- unattended migration runner must not hold cluster-admin rights over a
-- database it does not own. omninode_infra's own runner comment records the
-- same disposition: on RDS, "cluster-role provisioning is a provisioning-seam
-- concern (OMN-15343)".
--
-- So relocating the DDL to another migration loop — the OMN-16759 remedy for
-- the CREATE SCHEMA half of this class — is NOT available here: the node loop
-- connects as role_omnidash, which lacks CREATEROLE for the same reason. The
-- privilege is absent from every identity in the stream.
--
-- This file therefore ASSERTS the role and REFUSES to record itself when the
-- role is absent. It never reports success over a missing principal, which is
-- the masking outcome (OMN-14950) that would be strictly worse than the abort
-- it replaces: `src/omnibase_infra/topology/application_database.py` hard-binds
-- `tenant_projection` -> `tenant_projection_writer`, and OMN-16911 attests
-- current_user on every projection connection, so a silently-absent role
-- resurfaces as 100% DLQ on the tenant projections rather than as a failed
-- deploy. The role is provisioned at the seam that HOLDS the privilege —
-- omninode_infra `scripts/provision-cluster-roles.sh`, which reads the same
-- topology declaration this binding is resolved from — and this file is then
-- an honest no-op, exactly the disposition the OMN-15343 runner branch was
-- written to let through.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Role existence and attributes. The migration runner executes files through
--    asyncpg, so this must be regular SQL rather than psql meta-commands.
--    Every privileged command stays gated on a catalog read AND wrapped in a
--    handler that names the missing privilege (OMN-17301 D1).
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  executing_role text := current_user;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'tenant_projection_writer') THEN
    BEGIN
      CREATE ROLE tenant_projection_writer WITH NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
      RAISE NOTICE 'created role tenant_projection_writer as %', executing_role;
    EXCEPTION
      WHEN duplicate_object OR unique_violation THEN
        -- Roles are cluster-wide; two migration paths may race. Not an error.
        NULL;
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE = format(
            'tenant_projection_writer does not exist on this cluster and the '
            'executing role %I cannot create it: CREATE ROLE requires the '
            'CREATEROLE attribute, which every migration identity is '
            'deliberately provisioned without.', executing_role),
          DETAIL =
            'PostgreSQL roles are cluster-scoped. On the managed (RDS) lane the '
            'migrate Job holds only role_omnibase_infra and role_omnidash, both '
            'NOCREATEROLE by contract, and the instance has no superuser role '
            'this Job can authenticate as (OMN-15343). Relocating this DDL to '
            'the node migration loop does not help -- role_omnidash lacks the '
            'same attribute. This migration refuses to record itself against a '
            'principal that is not there: topology/application_database.py binds '
            'tenant_projection -> tenant_projection_writer and OMN-16911 attests '
            'current_user on every projection connection, so a silent skip would '
            'resurface as total DLQ loss on the tenant projections instead of as '
            'this message.',
          HINT =
            'Provision the role once at the seam that holds the privilege, then '
            're-run this deploy -- this file becomes an idempotent no-op. From '
            'omninode_infra, with the instance master credential in the '
            'environment: scripts/provision-cluster-roles.sh --apply '
            '(dry run by default; --help for the credential variables). '
            'Ticket: OMN-17301, class OMN-15343.';
    END;
  END IF;

  -- Attribute convergence. Each ALTER is gated on an OBSERVED divergence AND
  -- guarded: re-asserting an already-correct value is not idempotence, it is a
  -- privilege demand made on every apply (the defect that made 1.0.0
  -- undeliverable in the first place).
  IF EXISTS (
    SELECT 1
      FROM pg_catalog.pg_roles
     WHERE rolname = 'tenant_projection_writer'
       AND (rolcreatedb OR rolcreaterole)
  ) THEN
    BEGIN
      ALTER ROLE tenant_projection_writer NOCREATEDB NOCREATEROLE;
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE =
            'tenant_projection_writer carries CREATEDB and/or CREATEROLE and '
            'the executing role lacks the role-administration rights to remove '
            'them.',
          HINT =
            'Correct the role at the provisioning seam (omninode_infra '
            'scripts/provision-cluster-roles.sh --apply) before the tenant '
            'projection write path can be trusted. Ticket: OMN-17301.';
    END;
  END IF;

  -- NOSUPERUSER / NOBYPASSRLS / NOREPLICATION are the isolation controls this
  -- whole P5 cut exists to establish (OMN-14894 tenant_isolation RLS is
  -- enforced against exactly this principal), so an observed escalation is
  -- fatal whether or not it can be corrected here.
  IF EXISTS (
    SELECT 1
      FROM pg_catalog.pg_roles
     WHERE rolname = 'tenant_projection_writer'
       AND (rolsuper OR rolbypassrls OR rolreplication)
  ) THEN
    BEGIN
      ALTER ROLE tenant_projection_writer NOSUPERUSER NOBYPASSRLS NOREPLICATION;
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE =
            'tenant_projection_writer carries SUPERUSER, BYPASSRLS and/or '
            'REPLICATION and the executing role cannot remove them.',
          DETAIL =
            'Any of those flags silently exempts every tenant projection write '
            'from the RLS isolation this principal exists to be subject to.',
          HINT =
            'A role administrator must correct this at the provisioning seam '
            'before tenant projections may write. Ticket: OMN-17301.';
    END;
  END IF;
END
$$;

-- Fail if the role still does not exist. Unreachable when the block above
-- completed -- kept as a belt-and-braces assertion against a future edit that
-- weakens a handler into a swallow.
SELECT 'tenant_projection_writer'::regrole;

-- -----------------------------------------------------------------------------
-- 2. CONNECT on the target database (OMN-17301 D2/D3).
--
--    Two things were wrong here in 1.0.0 and both are fixed by READING BACK the
--    outcome instead of assuming it:
--
--    * The GRANT is issued by whichever role runs this file. On the managed
--      lane that is role_omnibase_infra, which does not own omnidash_analytics
--      (role_omnidash does) and therefore holds no grant option on it.
--      PostgreSQL does not raise for a no-op GRANT -- it emits
--      `WARNING: no privileges were granted` and returns success -- so the
--      statement "succeeded" while changing nothing.
--
--    * has_database_privilege(...,'CONNECT') cannot detect that, because CONNECT
--      is held by PUBLIC on any database that has not revoked it. The assertion
--      was true before this file ever ran.
--
--    What actually matters is that the principal CAN connect. That is asserted
--    below on its true terms, and the SOURCE of the privilege is reported, so
--    "explicitly granted" is never confused with "inherited from PUBLIC".
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  db_present        boolean;
  explicit_grant    boolean;
  effective_connect boolean;
BEGIN
  SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omnidash_analytics')
    INTO db_present;

  IF NOT db_present THEN
    RAISE NOTICE
      'omnidash_analytics is absent on this cluster; CONNECT grant skipped '
      '(this file stays valid on a cluster that has not been through 000).';
    RETURN;
  END IF;

  BEGIN
    GRANT CONNECT ON DATABASE omnidash_analytics TO tenant_projection_writer;
  EXCEPTION
    WHEN insufficient_privilege THEN
      RAISE NOTICE
        'GRANT CONNECT was refused for %; falling through to the readback, '
        'which decides whether the principal can connect regardless.',
        current_user;
  END;

  SELECT EXISTS (
    SELECT 1
      FROM pg_catalog.pg_database d
     WHERE d.datname = 'omnidash_analytics'
       AND d.datacl IS NOT NULL
       AND EXISTS (
             SELECT 1
               FROM aclexplode(d.datacl) a
              WHERE a.grantee = 'tenant_projection_writer'::regrole::oid
                AND a.privilege_type = 'CONNECT')
  ) INTO explicit_grant;

  SELECT has_database_privilege('tenant_projection_writer', 'omnidash_analytics', 'CONNECT')
    INTO effective_connect;

  IF NOT effective_connect THEN
    RAISE EXCEPTION USING
      ERRCODE = 'insufficient_privilege',
      MESSAGE =
        'tenant_projection_writer cannot CONNECT to omnidash_analytics and this '
        'migration could not grant it.',
      DETAIL = format(
        'The GRANT is issued by %I, which holds no grant option on that '
        'database on this cluster, and PUBLIC''s default CONNECT has been '
        'revoked, so nothing carries the privilege.', current_user),
      HINT =
        'Grant it at the provisioning seam: omninode_infra '
        'scripts/provision-cluster-roles.sh --apply. Ticket: OMN-17301.';
  END IF;

  IF explicit_grant THEN
    RAISE NOTICE
      'tenant_projection_writer holds CONNECT on omnidash_analytics by explicit grant.';
  ELSE
    RAISE NOTICE
      'tenant_projection_writer can CONNECT to omnidash_analytics via PUBLIC''s '
      'default grant; no explicit grant is recorded in datacl. This is '
      'sufficient for the binding and is NOT an error -- it is reported so that '
      '"granted" is never inferred from "can connect" (OMN-17301 D3).';
  END IF;
END
$$;
