-- Rollback: rollback_006_drop_skill_executions.sql
-- Purpose: Drop skill_executions table and associated indexes
-- Ticket: OMN-2934
-- WARNING: Destructive — all skill lifecycle observability data will be lost.

DROP TABLE IF EXISTS skill_executions CASCADE;
