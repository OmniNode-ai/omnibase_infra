-- Rollback: 029_create_db_metadata
-- Description: Drop db_metadata table (OMN-2085)

-- WARNING: This rollback drops the ownership marker for ALL services.
-- Only safe when rolling back the owning service's migrations.
DROP TABLE IF EXISTS public.db_metadata;
