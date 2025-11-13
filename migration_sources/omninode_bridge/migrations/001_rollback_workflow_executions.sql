-- Rollback: 001_create_workflow_executions
-- Description: Rollback workflow_executions table and indexes
-- Created: 2025-10-07

-- Drop indexes first
DROP INDEX IF EXISTS idx_workflow_executions_started_at;
DROP INDEX IF EXISTS idx_workflow_executions_state;
DROP INDEX IF EXISTS idx_workflow_executions_namespace;
DROP INDEX IF EXISTS idx_workflow_executions_correlation_id;

-- Drop table
DROP TABLE IF EXISTS workflow_executions CASCADE;

-- Note: We don't drop extensions as they might be used by other tables
-- If you need to drop extensions, run manually:
-- DROP EXTENSION IF EXISTS "pg_stat_statements";
-- DROP EXTENSION IF EXISTS "uuid-ossp";
