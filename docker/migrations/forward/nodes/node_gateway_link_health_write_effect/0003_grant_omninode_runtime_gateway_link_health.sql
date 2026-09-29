-- =============================================================================
-- MIGRATION: explicit omninode_runtime grants on omninode_internal.gateway_link_health
-- =============================================================================
-- Ticket: OMN-17886, AC2 step 2 (every declared grant delivered explicitly by a
-- migration in the lineage that owns the relation).
--
-- WHAT IS WRONG
--   HandlerGatewayLinkHealthUpsert runs INSERT ... ON CONFLICT (tenant_id) DO
--   UPDATE against this table as omninode_runtime, which needs SELECT, INSERT
--   and UPDATE. 0001 creates the table and grants nothing. The runtime's
--   privileges on it come only from the postgres-owned ALTER DEFAULT PRIVILEGES
--   rule in schema omninode_internal, on the lanes that have that rule. AC2
--   step 4 drops that rule, and on a lane without it the writer is denied:
--   OMN-15359's 2026-09-04 onex-dev readback found this table among the
--   omninode_internal tables omninode_runtime could not read.
--
-- WHAT THIS FILE DOES
--   It GRANTs exactly the privileges the topology now declares
--   (src/omnibase_infra/topology/table_grant_derivation.py, the
--   gateway_link_health supplemental entry, access read_write), and re-asserts
--   the declared USAGE on the schema, because a migration must not assume a
--   sibling file ran. It widens nothing beyond the declaration: no DELETE, as
--   the handler never deletes.
--
-- WHY THIS STREAM, AND 0003
--   0001 in this directory creates the table, so the grant sits next to it, the
--   convention node_projection_session_replay/0002 and
--   node_evidence_dashboard_reducer/0002 follow. 0002 in this stream is the
--   revoke on the gateway_link_health_status view (OMN-17886), which lands
--   separately; both runners apply a node directory in sorted order, so 0003
--   runs after it wherever both are present, and alone where it is not. 0001 is
--   not edited: it is applied on the lanes with a recorded content_sha256.
--
-- EXECUTING ROLE
--   GRANT by a role that is not the owner, a member of the owner role, a
--   superuser, or a holder of the privilege WITH GRANT OPTION grants nothing
--   and only raises a WARNING. So the file does not trust the GRANT: the
--   post-condition below reads the table's own ACL through aclexplode, and
--   fails the migration unless all three privileges are there for
--   omninode_runtime itself. information_schema is not used, because its grant
--   views show only grants involving a role the session belongs to. On a lane
--   where the runner cannot grant, the node loop stops and nothing is ledgered.
--
-- Idempotent: GRANT of privileges already held is a no-op, and so is the read.
-- Class: expand-only (GRANT only). Rollback:
-- rollback/rollback_node_gateway_link_health_0003.sql (manual).

GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;

GRANT SELECT, INSERT, UPDATE ON omninode_internal.gateway_link_health TO omninode_runtime;

-- Post-condition: division by zero unless the table's ACL carries SELECT,
-- INSERT and UPDATE for omninode_runtime.
SELECT 1 / (count(DISTINCT a.privilege_type) = 3)::int AS gateway_link_health_runtime_grant_assertion
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace,
     aclexplode(c.relacl) a
WHERE n.nspname = 'omninode_internal'
  AND c.relname = 'gateway_link_health'
  AND a.grantee = 'omninode_runtime'::regrole
  AND a.privilege_type IN ('SELECT', 'INSERT', 'UPDATE');
