-- =============================================================================
-- MIGRATION: revoke omninode_runtime's privileges on
--            omninode_internal.gateway_link_health_status (a VIEW)
-- =============================================================================
-- Ticket: OMN-17886, AC2 (omninode_internal live ACLs match the topology).
-- Ruling: operator decision 2026-09-25, RULING 2026-09-25T14:11:58Z, recorded on
--   OMN-17886 (comment 76652630): "revoke omninode_runtime privileges on
--   gateway_link_health_status".
--
-- WHAT IS WRONG
--   On a from-empty build of the corpus, omninode_runtime holds INSERT, SELECT
--   and UPDATE on this view. No migration grants them: they come from the
--   postgres-owned ALTER DEFAULT PRIVILEGES rule in schema omninode_internal,
--   which confers them on every relation created afterwards, 0001's view
--   included. The topology declares no grant on the view at all
--   (src/omnibase_infra/topology/table_grant_derivation.py, the comment beside
--   the gateway_link_health supplemental entry), so the live-ACL gate lists all
--   three as UNDECLARED_GRANT.
--
-- WHY REVOKE, NOT DECLARE
--   Nothing that runs as omninode_runtime reads or writes the view.
--   HandlerGatewayLinkHealthUpsert's only statement is an INSERT ... ON CONFLICT
--   into the table, omninode_internal.gateway_link_health. Searched on
--   2026-09-27 at the default branch of omnibase_infra, omnimarket, omnidash,
--   omniweb and omninode_infra: outside migrations, tests and comments the view
--   is named only by omninode_infra's k8s/migrations/
--   application-relation-ownership.yaml, whose entry for it lists
--   "operator SQL inspection" as its one reader and no writers. Declaring
--   privileges no runtime consumer uses would widen the runtime role for
--   nothing.
--
-- WHY THIS STREAM
--   0001 in this directory creates the view, so the node that owns it carries
--   the change to its ACL. The node loop is the path that reaches
--   omnidash_analytics (NODE_POSTGRES_DB in scripts/run-forward-migrations.sh).
--
-- CLASS
--   forward-only in config/migration_classes.yaml: REVOKE is on the class
--   checker's list of destructive statements
--   (scripts/validation/check_migration_class.py), so the file cannot be
--   expand-only, and it is not the destructive half of an expand/contract pair.
--
-- EXECUTING ROLE
--   A REVOKE issued by a role that is not the grantor (and not the owner, a
--   member of the owner role, or a superuser) removes nothing and only raises a
--   WARNING. So the file does not trust the REVOKE: the post-condition below
--   reads the view's own ACL. information_schema is not used for it, because
--   its grant views show only grants involving a role the session belongs to,
--   and an assertion that cannot see a grant would pass.
--
--   The consequence is deliberate and it is a deploy stop: on a lane where the
--   runtime role holds a grant the executing role cannot revoke, this file
--   fails, the node loop stops, and nothing is ledgered. That is the
--   fail-closed choice over ledgering a revoke that did not happen. Before this
--   reaches a lane, the lane's view ACL and its grantors are read back, read
--   only, so the stop is predicted rather than met.
--
-- SCOPE OF THE ASSERTION
--   It counts ACL entries whose grantee is omninode_runtime itself: the direct
--   grants the default-privilege rule confers, which is what the live-ACL gate
--   reports. Privileges reaching the role through PUBLIC or through membership
--   in another role are not grants this file made or can remove, and it does
--   not claim them.
--
-- LANES
--   From-empty build: INSERT, SELECT and UPDATE before this file, measured by
--   tests/integration/migrations/test_omninode_internal_live_acl_gate_omn17886.py
--   on PostgreSQL 16 on 2026-09-27. No live lane was read for this file. Where
--   the role holds nothing on the view, the REVOKE changes nothing and the
--   assertion passes. The paths, including a non-superuser runner and a foreign
--   grantor, are proved by
--   tests/integration/migrations/test_revoke_gateway_link_health_status_omn17886.py.
--
-- The default-privilege rule itself stays until OMN-17886 AC2 step 4, which
-- drops it last. A relation created before then still receives the rule's
-- grants; CREATE OR REPLACE VIEW keeps an existing ACL, so re-running 0001's
-- statement does not bring these privileges back. Dropping the view and
-- creating it again (rollback_node_gateway_link_health_0001.sql, then 0001)
-- does bring them back, and this file, already ledgered, does not run again;
-- the live-ACL gate reports that as UNDECLARED_GRANT until step 4.
--
-- Idempotent: a second REVOKE of privileges that are absent is a no-op.
-- Rollback: rollback/rollback_node_gateway_link_health_0002.sql (manual).

REVOKE ALL PRIVILEGES ON omninode_internal.gateway_link_health_status FROM omninode_runtime;

-- Post-condition: division by zero while the view's ACL still carries any
-- entry for omninode_runtime.
SELECT 1 / (count(*) = 0)::int AS gateway_link_health_status_runtime_revoked_assertion
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace,
     aclexplode(c.relacl) a
WHERE n.nspname = 'omninode_internal'
  AND c.relname = 'gateway_link_health_status'
  AND a.grantee = 'omninode_runtime'::regrole;
