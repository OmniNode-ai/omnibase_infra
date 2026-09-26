-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- Synthetic role stubs for the OMN-15356 fixture. The vendored
-- capability_scores tenant/RLS migration (0002) and the policy restatement
-- (0004) require app_dashboard to exist, be non-superuser, and be
-- non-bypassrls before they grant SELECT; the sequence grant (0004) requires
-- tenant_projection_writer. This harness proves the column-conversion stream
-- (0003 and its follow-ons), not the role-provisioning path (OMN-14899,
-- OMN-15655), so the roles are created directly rather than by vendoring
-- those migrations' own dependency chains.

\set ON_ERROR_STOP on

CREATE ROLE app_dashboard NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE tenant_projection_writer NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
