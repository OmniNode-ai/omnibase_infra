-- Rollback: rollback_006_drop_manifest_injection_lifecycle_table.sql
-- Reverses: 006_create_manifest_injection_lifecycle_table.sql
-- Ticket: OMN-2942

DROP TABLE IF EXISTS manifest_injection_lifecycle CASCADE;
