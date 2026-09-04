-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback: 104_create_validator_ro_role (OMN-17792)
--
-- Revokes validator_ro's read grants in omnidash_analytics (issued by the
-- node-owned companion, not by 104 -- `DROP ROLE` fails while any grant to the
-- role remains), revokes CONNECT, and drops the role. Manual/operator-run, like
-- all rollbacks. This file is NOT applied by any runner and is deliberately
-- allowed the cross-database `\connect` that a forward flat migration is not:
-- the OMN-15819 gate scopes to `docker/migrations/forward/*.sql`, because the
-- k8s Job's flat loop is what cannot deliver such a file.
--
-- Rolling this back removes the scoped read identity, which returns database
-- validation on the onex-dev RDS to `role_omnidash` -- the migration principal
-- that OWNS its relations, and is therefore exempt from row-level security and
-- able to DROP/ALTER/TRUNCATE them. That is a widening of privilege, not a
-- cleanup. Do not run it as routine hygiene.
--
-- The credential attach is NOT undone here. It is deployment-owned (the
-- OMN-16843 / OMN-17733 pattern) and its value lives only in the secret store;
-- retiring it is a store operation, and `DROP ROLE` below makes the credential
-- inert regardless.

\connect omnidash_analytics

DO $$
DECLARE
  target_schema text;
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'validator_ro') THEN
    FOREACH target_schema IN ARRAY ARRAY[
      'public',
      'omninode_internal',
      'platform_catalog',
      'tenant'
    ] LOOP
      IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = target_schema) THEN
        EXECUTE format(
          'ALTER DEFAULT PRIVILEGES IN SCHEMA %I REVOKE SELECT ON TABLES FROM validator_ro',
          target_schema);
        EXECUTE format(
          'REVOKE ALL ON ALL TABLES IN SCHEMA %I FROM validator_ro', target_schema);
        EXECUTE format(
          'REVOKE USAGE ON SCHEMA %I FROM validator_ro', target_schema);
      END IF;
    END LOOP;
  END IF;
END;
$$;

\connect omnibase_infra

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'validator_ro')
     AND EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnidash_analytics') THEN
    EXECUTE 'REVOKE CONNECT ON DATABASE omnidash_analytics FROM validator_ro';
  END IF;
END;
$$;

DROP ROLE IF EXISTS validator_ro;
