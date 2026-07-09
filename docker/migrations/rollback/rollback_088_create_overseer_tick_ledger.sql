-- Rollback: rollback_088_create_overseer_tick_ledger.sql
-- Reverts:  088_create_overseer_tick_ledger.sql
-- Ticket:   OMN-13996 (child of epic OMN-13989)
--
-- Drops the overseer_tick_ledger append-only projection table and its indexes.
-- Indexes are dropped implicitly with the table; listed for clarity.

DROP TABLE IF EXISTS overseer_tick_ledger CASCADE;
