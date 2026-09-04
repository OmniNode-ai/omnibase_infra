-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- =============================================================================
-- MIGRATION 104: Create the validator_ro read-only role (OMN-17792)
-- =============================================================================
-- Ticket: OMN-17792 (AC6, the RDS half). Class: OMN-17301 (D1/D2/D3),
--         OMN-15343 (the provisioning seam), OMN-15819 (the
--         undeliverable-flat-file class this file is written to avoid joining).
-- Version: 1.0.0
--
-- =============================================================================
-- WHY THIS FILE EXISTS
-- =============================================================================
-- OMN-17792 AC6 asks for a read-only PostgreSQL role that makes the database
-- half of OMN-17298, OMN-15359, OMN-15425 and the OMN-17440 grant residual
-- answerable. A role was provisioned first on the `.201` dev lane, but that is
-- not the lane those four tickets validate against: they run against the
-- `onex-dev` serving RDS `omninode-dev-postgres` (physical database
-- `omnidash_analytics`), reached through the dev-system cluster EC2
-- `i-06169517a92b45f86`.
--
-- The read identity in use on that instance today is `role_omnidash`, which is
-- the RDS migration principal under the OMN-15335 two-owner split and OWNS its
-- relations -- live-verified 21 owned relations (11 in `omninode_internal`, 10
-- in `public`) as of 2026-09-04. Ownership means DROP, ALTER and TRUNCATE, and
-- Postgres exempts a relation's owner from row-level security unconditionally,
-- FORCE included.
--
-- So the argument for this role is NOT that access is missing. It is that the
-- access being used for validation is far wider than validation requires, and
-- that a scoped principal makes destructive action impossible rather than
-- merely unintended -- while `NOBYPASSRLS` keeps the validator subject to the
-- very isolation it is validating.
--
-- The role is named after its PURPOSE, not after the person holding it today.
-- This file, the topology and the projections are shared artifacts; a person's
-- name in them is a rename waiting to happen and a re-grant nobody audits.
--
-- =============================================================================
-- WHAT THIS FILE DOES, AND WHAT IT DELIBERATELY DOES NOT
-- =============================================================================
-- DOES: assert the cluster-wide role, enforce the RLS-relevant attributes off
-- on every run, grant CONNECT on `omnidash_analytics`, and READ BACK whether
-- the privilege is carried by an explicit grant or by PUBLIC's default.
--
-- DOES NOT carry credential material and does NOT grant LOGIN. LOGIN + password
-- is a deployment-owned attach BY REFERENCE (the OMN-16843 / OMN-17733 pattern
-- `scripts/attach-topology-login-credential.sh`); the value lands in the secret
-- store and the holder is given ACCESS to the store path, never the value.
--
-- DOES NOT re-assert NOLOGIN on a pre-existing role. 094's invariant, unchanged:
-- re-asserting it would REVOKE that deployment-owned attach as a side effect of
-- recording a migration. NOLOGIN is a CREATE-time default here and nothing else.
-- LOGIN is not the isolation control in any case -- NOSUPERUSER, NOBYPASSRLS and
-- non-ownership are, and all three are enforced on every run.
--
-- DOES NOT grant schema USAGE or any relation privilege. See the next section;
-- this is the half of the design that has gone wrong twice before.
--
-- =============================================================================
-- WHY NO SCHEMA OR TABLE GRANT HERE -- AND WHY THAT IS NOT AN OMISSION
-- =============================================================================
-- Two files in this corpus already tried to carry the per-database half of
-- exactly this shape, and both are tombstoned:
--
--   * 096 (`role_omnidash` authorization) and 097 (`app_dashboard` CONNECT +
--     schema USAGE) each `\connect omnidash_analytics` partway through. The k8s
--     Job that applies this corpus (`omninode_infra`,
--     `k8s/migrations/omnibase-infra-migrate.yaml`) owns exactly one database
--     and gates its flat `psql -f` loop on `directive_db == "$DB_NAME"`, where
--     `DB_NAME` is `omnibase_infra`. Both files were therefore UNREACHABLE on
--     the managed lane, in that loop or any other in the runner -- and both
--     accreted a false "applied" ledger row that hid it (OMN-15819 / OMN-15846).
--     097's `GRANT USAGE ON SCHEMA public TO app_dashboard` left no trace on
--     RDS: that grant is recorded as issued by `pg_database_owner`, never by
--     the identity this file would have executed under.
--
-- A NEW cross-database flat file is now a hard, fail-closed reject
-- (`tests/ci/test_flat_migration_no_foreign_connect_gate.py` /
-- `scripts/ci/check_flat_migration_foreign_connect.py`; the manifest is a
-- CLOSED ledger frozen to the five filenames that predate the gate). Adding an
-- entry to that manifest is not a way to authorise a new one.
--
-- The per-database half therefore rides the NODE-OWNED loop, which connects
-- directly to `omnidash_analytics` as `role_omnidash` (`NODE_POSTGRES_DB`) --
-- the same seam OMN-15425 used for `tenant_projection_writer`
-- (`docker/migrations/forward/nodes/node_projection_delegation_inference_response/
-- 0004_grant_tenant_projection_writer.sql`). That loop's files are VENDORED from
-- omnimarket by `scripts/sync-node-migrations.sh`, so the grant file is authored
-- there and vendored here; a file authored only in this repo is reported as
-- stale drift and removed by the next sync. That companion is tracked as the
-- OMN-17792 node-grant follow-up and is what delivers:
--
--     GRANT USAGE ON SCHEMA public, omninode_internal, platform_catalog
--       (and tenant, guarded on existence) TO validator_ro;
--     GRANT SELECT ON ALL TABLES IN SCHEMA ... TO validator_ro;
--     ALTER DEFAULT PRIVILEGES ... GRANT SELECT ON TABLES TO validator_ro;
--
-- Until it lands, `validator_ro` can open a session and read nothing. That is
-- stated here, and executed in `tests/unit/db/test_migration_104_omn17792.py`,
-- so "the read-only role migration landed" is never mistaken for "the role can
-- read". A partially-provisioned principal that looks finished is the failure
-- mode this section exists to prevent.
--
-- =============================================================================
-- WHY THE ROLE IS NOT CREATED HERE ON THE MANAGED LANE
-- =============================================================================
-- PostgreSQL roles are CLUSTER-scoped and `CREATE ROLE` requires CREATEROLE. The
-- migrate Job holds exactly two credentials against the managed instance --
-- `role_omnibase_infra` (flat loop) and `role_omnidash` (node loop) -- and BOTH
-- are provisioned NOCREATEROLE by contract (`omninode_infra`
-- `scripts/init-databases.sh`: "no CREATEDB, no SUPERUSER, no CREATEROLE"). The
-- instance has no `postgres` role at all; its master credential lives in AWS
-- Secrets Manager under terraform `manage_master_user_password` and is
-- deliberately not mounted into an unattended Job (OMN-15335). Relocating the
-- DDL to the node loop does not help -- `role_omnidash` lacks the same
-- attribute. The privilege is absent from every identity in the stream.
--
-- So this file ASSERTS the role and REFUSES to record itself when it is absent,
-- naming the seam that holds the privilege. It never reports success over a
-- missing principal, which is the OMN-14950 masking outcome and is strictly
-- worse than an abort. On the normal path the abort is unreachable: the
-- "Provision topology cluster roles (OMN-17347)" step of
-- `deploy-onex-staging.yml` runs `provision-cluster-roles.sh --apply` before the
-- migration stream on every deploy, and that script reads the SAME topology
-- `principals` block this role is declared in -- so a principal added to the
-- topology is created, and granted CONNECT, with no edit to any script.
--
-- =============================================================================
-- DESIGN INVARIANTS
-- =============================================================================
--   * NOSUPERUSER + NOBYPASSRLS are ENFORCED on every run, not merely requested
--     at create time. `validator_ro` exists to be SUBJECT to the OMN-14894
--     tenant_isolation policies; either flag would silently exempt it and make
--     every "the database reads clean" finding taken through it worthless.
--   * `validator_ro` must NEVER own a relation. Nothing here grants CREATE on a
--     database or a schema, and the companion node file grants none either.
--   * EVERY privileged statement is gated on an OBSERVED DIVERGENCE. `ALTER
--     ROLE` demands role-administration rights, and SUPERUSER / BYPASSRLS /
--     REPLICATION are additionally gated by the EXECUTING role's own attributes
--     -- even to re-assert an already-correct `false`. An unconditional ALTER is
--     not idempotent; it is a privilege demand made on every apply, and it is
--     what made migration 103 v1.0.0 undeliverable (OMN-17301).
--   * The CONNECT outcome is READ BACK, never assumed (OMN-17301 D2/D3). A GRANT
--     issued by a role holding no grant option on the target database does not
--     raise -- PostgreSQL emits `WARNING: no privileges were granted` and
--     returns success -- and `has_database_privilege(...,'CONNECT')` is TRUE via
--     PUBLIC's default on any database that has not revoked it, so it cannot
--     detect that. The readback distinguishes the two and says which one carries
--     the privilege.
--
-- IDEMPOTENCY:
--   Safe to re-run. The CREATE is skipped when pg_roles already shows the role
--   and keeps the duplicate_object / unique_violation guard for the genuine
--   race (roles are cluster-wide; two migration paths may race). Every ALTER
--   runs only on an observed divergence, so a correct role touches none of them.
--   GRANT is idempotent in PostgreSQL by construction.
--
-- ROLLBACK:
--   See rollback/rollback_104_create_validator_ro_role.sql. It is operator-run,
--   is not applied by any runner, and widens privilege rather than cleaning up:
--   removing this role returns RDS validation to role_omnidash, which owns its
--   relations and is therefore RLS-exempt.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Role existence and attributes. The runner executes these files through
--    asyncpg on some lanes, so this is regular SQL rather than psql
--    meta-commands. Every privileged command is gated on a catalog read AND
--    wrapped in a handler that names the missing privilege.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  executing_role text := current_user;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'validator_ro') THEN
    BEGIN
      CREATE ROLE validator_ro WITH NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
      RAISE NOTICE 'created role validator_ro as %', executing_role;
    EXCEPTION
      WHEN duplicate_object OR unique_violation THEN
        -- Roles are cluster-wide; two migration paths may race. Not an error.
        NULL;
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE = format(
            'validator_ro does not exist on this cluster and the executing role '
            '%I cannot create it: CREATE ROLE requires the CREATEROLE attribute, '
            'which every migration identity is deliberately provisioned without.',
            executing_role),
          DETAIL =
            'PostgreSQL roles are cluster-scoped. On the managed (RDS) lane the '
            'migrate Job holds only role_omnibase_infra and role_omnidash, both '
            'NOCREATEROLE by contract, and the instance has no superuser role '
            'this Job can authenticate as (OMN-15343). Relocating this DDL to the '
            'node migration loop does not help -- role_omnidash lacks the same '
            'attribute. This migration refuses to record itself against a '
            'principal that is not there: a silently-absent role would resurface '
            'as a validator that reads zero rows and reports the database empty, '
            'instead of as this message.',
          HINT =
            'Provision the role once at the seam that holds the privilege, then '
            're-run this deploy -- this file becomes an idempotent no-op. The '
            'deploy pipeline does this automatically in the "Provision topology '
            'cluster roles (OMN-17347)" step of deploy-onex-staging.yml; by hand, '
            'from omninode_infra with the instance master credential in the '
            'environment: scripts/provision-cluster-roles.sh --apply --instance '
            'onex-dev (dry run by default; --help for the credential variables). '
            'Ticket: OMN-17792, class OMN-15343.';
    END;
  END IF;

  -- CREATEDB / CREATEROLE. Gated on an observed divergence: ALTER ROLE demands
  -- role-administration rights even when the value it sets is already correct.
  IF EXISTS (
    SELECT 1
      FROM pg_catalog.pg_roles
     WHERE rolname = 'validator_ro'
       AND (rolcreatedb OR rolcreaterole)
  ) THEN
    BEGIN
      ALTER ROLE validator_ro NOCREATEDB NOCREATEROLE;
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE =
            'validator_ro carries CREATEDB and/or CREATEROLE and the executing '
            'role lacks the role-administration rights to remove them.',
          HINT =
            'Correct the role at the provisioning seam (omninode_infra '
            'scripts/provision-cluster-roles.sh --apply --instance onex-dev) '
            'before any finding is taken through it. Ticket: OMN-17792.';
    END;
  END IF;

  -- NOSUPERUSER / NOBYPASSRLS / NOREPLICATION are the reason this role exists in
  -- the shape it does, so an observed escalation is FATAL whether or not it can
  -- be corrected from here.
  IF EXISTS (
    SELECT 1
      FROM pg_catalog.pg_roles
     WHERE rolname = 'validator_ro'
       AND (rolsuper OR rolbypassrls OR rolreplication)
  ) THEN
    BEGIN
      ALTER ROLE validator_ro NOSUPERUSER NOBYPASSRLS NOREPLICATION;
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE =
            'validator_ro carries SUPERUSER, BYPASSRLS and/or REPLICATION and '
            'the executing role cannot remove them.',
          DETAIL =
            'Any of those flags exempts this principal from the tenant_isolation '
            'row-level security it exists to be subject to. A validator that can '
            'bypass RLS reads the rows the policies withhold and reports the '
            'database clean -- masking the exact defect class OMN-17298 and '
            'OMN-17422 turn on.',
          HINT =
            'A role administrator must correct this at the provisioning seam '
            'before this principal may be used for validation. Ticket: OMN-17792.';
    END;
  END IF;
END
$$;

-- Fail if the role still does not exist. Unreachable when the block above
-- completed -- kept as a belt-and-braces assertion against a future edit that
-- weakens a handler into a swallow.
SELECT 'validator_ro'::regrole;

-- -----------------------------------------------------------------------------
-- 2. CONNECT on the target database, with the outcome READ BACK.
--
--    `GRANT ... ON DATABASE` is cluster-wide, so it is issued from the current
--    (`omnibase_infra`) context and needs no `\connect` -- which is the whole
--    reason this file is deliverable where 096 and 097 are not.
--
--    Both OMN-17301 defects are guarded here:
--      D2  the GRANT is issued by whichever role runs this file. On the managed
--          lane that is role_omnibase_infra, which does not own
--          omnidash_analytics (role_omnidash does) and holds no grant option on
--          it. PostgreSQL does not raise for that -- it emits
--          `WARNING: no privileges were granted` and returns success.
--      D3  has_database_privilege(...,'CONNECT') cannot detect D2, because
--          CONNECT is held by PUBLIC on any database that has not revoked it.
--
--    What matters is that the principal CAN connect. That is asserted on its
--    true terms below, and the SOURCE of the privilege is reported, so
--    "explicitly granted" is never confused with "inherited from PUBLIC" -- a
--    distinction that becomes load-bearing the day OMN-15355 revokes PUBLIC's
--    default.
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
    GRANT CONNECT ON DATABASE omnidash_analytics TO validator_ro;
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
              WHERE a.grantee = 'validator_ro'::regrole::oid
                AND a.privilege_type = 'CONNECT')
  ) INTO explicit_grant;

  SELECT has_database_privilege('validator_ro', 'omnidash_analytics', 'CONNECT')
    INTO effective_connect;

  IF NOT effective_connect THEN
    RAISE EXCEPTION USING
      ERRCODE = 'insufficient_privilege',
      MESSAGE =
        'validator_ro cannot CONNECT to omnidash_analytics and this migration '
        'could not grant it.',
      DETAIL = format(
        'The GRANT is issued by %I, which holds no grant option on that database '
        'on this cluster, and PUBLIC''s default CONNECT has been revoked, so '
        'nothing carries the privilege.', current_user),
      HINT =
        'Grant it at the provisioning seam: omninode_infra '
        'scripts/provision-cluster-roles.sh --apply --instance onex-dev. '
        'Ticket: OMN-17792.';
  END IF;

  IF explicit_grant THEN
    RAISE NOTICE
      'validator_ro holds CONNECT on omnidash_analytics by explicit grant.';
  ELSE
    RAISE NOTICE
      'validator_ro can CONNECT to omnidash_analytics via PUBLIC''s default '
      'grant; no explicit grant is recorded in datacl. This is sufficient for '
      'the role today and is NOT an error -- it is reported so that "granted" is '
      'never inferred from "can connect" (OMN-17301 D3), and so the day '
      'OMN-15355 revokes PUBLIC''s CONNECT this role is known to need an '
      'explicit grant from the provisioning seam.';
  END IF;
END
$$;
