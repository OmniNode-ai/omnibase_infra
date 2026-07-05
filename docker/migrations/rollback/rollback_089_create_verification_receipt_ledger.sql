-- Rollback: rollback_089_create_verification_receipt_ledger.sql
-- Reverts:  089_create_verification_receipt_ledger.sql
-- Ticket:   OMN-13997 (child of epic OMN-13989)
--
-- Drops the verification_receipt_ledger append-only projection table and its
-- indexes. Indexes are dropped implicitly with the table.

DROP TABLE IF EXISTS verification_receipt_ledger CASCADE;
