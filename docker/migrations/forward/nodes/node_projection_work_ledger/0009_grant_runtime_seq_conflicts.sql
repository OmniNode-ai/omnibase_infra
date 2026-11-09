-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- =============================================================================
-- MIGRATION: grant the runtime role its set on work_ledger_seq_conflicts
-- =============================================================================
-- Ticket: OMN-20413. The omnimarket node_projection_work_ledger projection
-- binding (omnimarket#5466) writes omninode_internal.work_ledger_seq_conflicts,
-- and the runtime refuses to wire a binding whose principal lacks SELECT,
-- INSERT and UPDATE on it. The relation is created by the omnimarket change,
-- so the grant is guarded on the relation existing: a database that applies
-- this chain before that table exists skips the grant instead of failing every
-- migration behind it. DELETE is deliberately not granted.
-- =============================================================================

DO $$
BEGIN
    IF to_regclass('omninode_internal.work_ledger_seq_conflicts') IS NOT NULL THEN
        GRANT SELECT, INSERT, UPDATE
            ON TABLE omninode_internal.work_ledger_seq_conflicts
            TO omninode_runtime;
    END IF;
END
$$;
