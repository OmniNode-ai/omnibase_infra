-- Rollback Migration: 011_projection_and_watermarks
-- Description: Drop projection store and watermark tracking tables
-- Created: 2025-10-21

-- Drop indexes first (workflow_projection)
DROP INDEX IF EXISTS idx_projection_indices_gin;
DROP INDEX IF EXISTS idx_projection_version;
DROP INDEX IF EXISTS idx_projection_namespace_tag;
DROP INDEX IF EXISTS idx_projection_tag;
DROP INDEX IF EXISTS idx_projection_namespace;

-- Drop tables
DROP TABLE IF EXISTS projection_watermarks;
DROP TABLE IF EXISTS workflow_projection;
