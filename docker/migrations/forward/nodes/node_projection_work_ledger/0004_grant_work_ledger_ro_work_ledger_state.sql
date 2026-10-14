-- =============================================================================
-- MIGRATION: grant the work-ledger parity check's read-only role the state table
-- =============================================================================
-- Ticket:  OMN-20737. Follows 0001 (the runtime role's write set).
-- Owner:   omnibase_infra, infra-first: this lands ahead of any omnimarket
--          source file, the same window 0001 and 0003 used.
--
-- work_ledger_ro is a NOSUPERUSER NOBYPASSRLS login that exists only on the
-- dev lane's omnidash_analytics; no migration creates it, and no
-- credential is created, changed or read here. It already holds SELECT on
-- work_ledger_rows and USAGE on omninode_internal. The parity check reads
-- work_ledger_state too and exits 2 on "permission denied for table
-- work_ledger_state" until this grant lands.
--
-- The grant is guarded on the role existing: every other database that applies
-- this chain (a fresh compose volume, a pre-PR slot, the other lanes, prod)
-- has no work_ledger_ro, and an unguarded GRANT there fails the file and
-- blocks every migration behind it. SELECT only: the role never writes.
-- =============================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'work_ledger_ro') THEN
        GRANT SELECT ON TABLE omninode_internal.work_ledger_state TO work_ledger_ro;
    END IF;
END
$$;
