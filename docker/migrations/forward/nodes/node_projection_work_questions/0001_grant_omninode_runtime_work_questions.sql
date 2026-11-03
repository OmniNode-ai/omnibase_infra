-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-22320: runtime grants for the five work-question tables of 0000.
-- Owner: omnimarket.nodes.node_projection_work_questions
-- Target database: omnidash_analytics; physical schema: omninode_internal.
--
-- The writer reads the stored generation and upserts the newer one through the
-- runtime role, so the role gets SELECT, INSERT and UPDATE and never DELETE (a
-- generation leaves by the reducer's retention). Migration 099's schema
-- defaults already confer these on a database built from empty; this file
-- states them in the node's own lineage so a database built any other way gets
-- the same ACL, and asserts each one. The tables have no sequence.

GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;

GRANT SELECT, INSERT, UPDATE
    ON omninode_internal.work_outcome_units
    TO omninode_runtime;

GRANT SELECT, INSERT, UPDATE
    ON omninode_internal.orchestration_style_outcomes
    TO omninode_runtime;

GRANT SELECT, INSERT, UPDATE
    ON omninode_internal.lane_drift_and_stalls
    TO omninode_runtime;

GRANT SELECT, INSERT, UPDATE
    ON omninode_internal.brief_quality_vs_outcome
    TO omninode_runtime;

GRANT SELECT, INSERT, UPDATE
    ON omninode_internal.subagent_waste
    TO omninode_runtime;

SELECT 1 / count(*) AS work_outcome_units_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'work_outcome_units'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS work_outcome_units_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'work_outcome_units'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS work_outcome_units_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'work_outcome_units'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'UPDATE';

SELECT 1 / count(*) AS orchestration_style_outcomes_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'orchestration_style_outcomes'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS orchestration_style_outcomes_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'orchestration_style_outcomes'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS orchestration_style_outcomes_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'orchestration_style_outcomes'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'UPDATE';

SELECT 1 / count(*) AS lane_drift_and_stalls_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'lane_drift_and_stalls'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS lane_drift_and_stalls_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'lane_drift_and_stalls'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS lane_drift_and_stalls_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'lane_drift_and_stalls'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'UPDATE';

SELECT 1 / count(*) AS brief_quality_vs_outcome_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'brief_quality_vs_outcome'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS brief_quality_vs_outcome_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'brief_quality_vs_outcome'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS brief_quality_vs_outcome_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'brief_quality_vs_outcome'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'UPDATE';

SELECT 1 / count(*) AS subagent_waste_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'subagent_waste'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS subagent_waste_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'subagent_waste'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS subagent_waste_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'subagent_waste'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'UPDATE';
