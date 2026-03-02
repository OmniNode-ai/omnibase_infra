-- Rollback 033: Drop db_error_tickets table and indexes
--
-- Reverses migration 033_create_db_error_tickets.sql

DROP INDEX IF EXISTS idx_db_error_tickets_last_seen_at;
DROP INDEX IF EXISTS idx_db_error_tickets_fingerprint;
DROP TABLE IF EXISTS db_error_tickets;
