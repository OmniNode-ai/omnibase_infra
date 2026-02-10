-- Rollback: 001_registration_projection.sql
-- Purpose: Drop registration projection table and enum type
--
-- WARNING: This migration is DESTRUCTIVE. All data in registration_projections will be lost.
-- Consider backing up data before rollback:
--   COPY registration_projections TO '/tmp/backup.csv' CSV HEADER;

DROP TABLE IF EXISTS registration_projections CASCADE;
DROP TYPE IF EXISTS registration_state CASCADE;
