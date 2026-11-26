-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- The declared migration path for a cloud database (OMN-26712).
--
-- One command, run as the database's owner against the cloud Postgres, from the migrate
-- image's /migrations/seams/cloud directory (the image carries seams/ whole):
--
--   psql -X -q -v ON_ERROR_STOP=1 -h $HOST -U $OWNER -d $ONEX_CLOUD_DATABASE \
--        -f /migrations/seams/cloud/apply_cloud_database.sql
--
-- SETTINGS (environment; nothing here names a lab role, database or address):
--   ONEX_CLOUD_DATABASE                       the cloud database to migrate (required)
--   ONEX_CLOUD_GATEWAY_ROLE                   the gateway's login role (required)
--   ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE    the tenant identity writer's login role (required)
-- Both roles are created by the cloud owner, who also stores their credentials; this path
-- issues no credential and no role. A missing setting or a missing role FAILS the run.
--
-- IDEMPOTENT: the schema step creates a table only where it is absent and never alters one
-- (tenant_identity_schema.sql), the expiry step is ADD COLUMN IF NOT EXISTS, the readback
-- fails on any drift, and each grant step revokes then re-grants its exact list inside one
-- transaction. A second run against the same database changes nothing.

\set cloud_database ''
\set gateway_role ''
\set writer_role ''
\getenv cloud_database ONEX_CLOUD_DATABASE
\getenv gateway_role ONEX_CLOUD_GATEWAY_ROLE
\getenv writer_role ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE

SELECT
    :'cloud_database' = '' AS no_database,
    :'gateway_role' = '' AS no_gateway_role,
    :'writer_role' = '' AS no_writer_role
\gset

\if :no_database
  \echo [cloud-migration] FAIL ONEX_CLOUD_DATABASE is not set
  DO $cloud_setting$ BEGIN RAISE EXCEPTION 'ONEX_CLOUD_DATABASE is not set'; END $cloud_setting$;
\endif
\if :no_gateway_role
  \echo [cloud-migration] FAIL ONEX_CLOUD_GATEWAY_ROLE is not set
  DO $cloud_setting$ BEGIN RAISE EXCEPTION 'ONEX_CLOUD_GATEWAY_ROLE is not set'; END $cloud_setting$;
\endif
\if :no_writer_role
  \echo [cloud-migration] FAIL ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE is not set
  DO $cloud_setting$ BEGIN RAISE EXCEPTION 'ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE is not set'; END $cloud_setting$;
\endif

\set cloud_start_db :DBNAME
\ir ../tenant_identity_schema.sql
\ir ../tenant_api_keys_expiry.sql
\ir ../tenant_identity_schema_readback.sql
\ir gateway_grants.sql
\ir tenant_identity_writer_grants.sql
\connect :"cloud_start_db"
\echo [cloud-migration] ok    :cloud_database migrated; gateway and tenant identity writer grants asserted
