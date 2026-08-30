-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback: 103_create_tenant_projection_writer_role (OMN-15425)
--
-- Revokes tenant_projection_writer's grants in omnidash_analytics (issued by
-- the node-owned companion
-- nodes/node_projection_delegation_inference_response/0004_grant_tenant_projection_writer.sql
-- — DROP ROLE fails while any grant to the role remains), revokes CONNECT, and
-- drops the role. Manual/operator-run, like all rollbacks.
--
-- Rolling this back re-opens the outage this migration closed: every tenant
-- projection binding fails its OMN-16911 identity attestation and DLQs 100% of
-- its input. Do not run it as routine cleanup.

\connect omnidash_analytics

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tenant_projection_writer') THEN
    EXECUTE 'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM tenant_projection_writer';
    EXECUTE 'REVOKE USAGE ON SCHEMA public FROM tenant_projection_writer';
    IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'tenant') THEN
      EXECUTE 'REVOKE ALL ON ALL TABLES IN SCHEMA tenant FROM tenant_projection_writer';
      EXECUTE 'REVOKE USAGE ON SCHEMA tenant FROM tenant_projection_writer';
    END IF;
  END IF;
END;
$$;

\connect omnibase_infra

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tenant_projection_writer')
     AND EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnidash_analytics') THEN
    EXECUTE 'REVOKE CONNECT ON DATABASE omnidash_analytics FROM tenant_projection_writer';
  END IF;
END;
$$;

DROP ROLE IF EXISTS tenant_projection_writer;
