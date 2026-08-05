-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- Synthetic role stub. The vendored capability_scores tenant/RLS migration
-- (0002) requires app_dashboard to exist, be non-superuser, and be
-- non-bypassrls before it grants SELECT -- this harness proves the OMN-15356
-- column-conversion migration (0003), not the role-provisioning path
-- (OMN-14899), so the role is created directly rather than by vendoring that
-- migration's own dependency chain.

\set ON_ERROR_STOP on

CREATE ROLE app_dashboard NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
