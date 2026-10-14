-- =============================================================================
-- MIGRATION: let the read-only parity role read omninode_internal.work_ledger_state
-- =============================================================================
-- Ticket:  OMN-20737
-- Owner:   omnimarket.nodes.node_projection_work_ledger
--
-- The ledger parity check (node_projection_work_ledger.parity) reads both
-- work_ledger_rows and work_ledger_state. The dev lane's read-only role
-- work_ledger_ro was provisioned with SELECT on work_ledger_rows only, so the
-- check died on work_ledger_state with InsufficientPrivilegeError (exit 1, no
-- counts). This grants the one missing privilege.
--
-- SELECT only, on this one table: no INSERT, UPDATE or DELETE, no grant option,
-- no other relation. Nothing here creates the role, alters an attribute of it
-- or touches its credential.
--
-- work_ledger_ro is provisioned out of band on the lab dev database; no
-- migration creates it, so a fresh build (CI, local) has no such role and a bare
-- GRANT would abort with 42704. The grant is therefore guarded on the role
-- existing, and when it does exist the block proves the privilege landed.
-- GRANT is idempotent: re-running changes nothing.
-- =============================================================================

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'work_ledger_ro') THEN
    GRANT SELECT ON TABLE omninode_internal.work_ledger_state TO work_ledger_ro;

    IF NOT has_table_privilege('work_ledger_ro', 'omninode_internal.work_ledger_state', 'SELECT') THEN
      RAISE EXCEPTION 'work_ledger_ro cannot read omninode_internal.work_ledger_state after the grant';
    END IF;
  END IF;
END
$$;
