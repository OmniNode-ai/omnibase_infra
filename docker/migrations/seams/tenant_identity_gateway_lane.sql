-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- The lane gate for the tenant identity schema seams (OMN-26121).
--
-- Included first by gateway_runtime_grants.sql, which scripts/run-forward-migrations.sh
-- applies inside its slot-guarded grant seam (never under a slot). The runner is a baselined
-- file and cannot grow, so the lane `case` lives here: psql reads the lane indicator itself.
--
--   dev | stability_test  run tenant_identity_schema.sql, then tenant_api_keys_expiry.sql,
--                         then tenant_identity_schema_readback.sql (all three in this
--                         directory), then return to the database the session started on.
--   unset                 named skip: this lane does not serve the gateway.
--   anything else         a warning and the same skip (fail closed).
--
-- The indicator is set in docker-compose.dev-lane.yml and the stability lane overlay only,
-- never in docker-compose.infra.yml, the catalog or the collaborator lane.

\set gateway_lane ''
\getenv gateway_lane ONEX_MIGRATION_GATEWAY_LANE
SELECT
    :'gateway_lane' IN ('dev', 'stability_test') AS gateway_lane_ok,
    :'gateway_lane' = '' AS gateway_lane_unset
\gset

\if :gateway_lane_ok
  \set identity_start_db :DBNAME
  \ir tenant_identity_schema.sql
  \ir tenant_api_keys_expiry.sql
  \ir tenant_identity_schema_readback.sql
  \connect :identity_start_db
\elif :gateway_lane_unset
  \echo [forward-migration]   skip  tenant identity schema (ONEX_MIGRATION_GATEWAY_LANE unset: this lane does not serve the gateway)
\else
  \warn [forward-migration] WARNING: unknown ONEX_MIGRATION_GATEWAY_LANE value :gateway_lane, skipping the tenant identity schema (fail-closed).
\endif
