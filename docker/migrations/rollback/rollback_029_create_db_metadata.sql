-- Rollback: 029_create_db_metadata
-- Description: Drop db_metadata table (OMN-2085)

DROP TABLE IF EXISTS public.db_metadata;
