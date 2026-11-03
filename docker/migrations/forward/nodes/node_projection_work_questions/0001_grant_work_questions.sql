-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-22258: explicitly deliver the declared work-question runtime privileges.
-- These must survive removal of migration 099's default-privilege rule.
BEGIN;

GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.work_outcome_units TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.orchestration_style_outcomes TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.lane_drift_and_stalls TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.brief_quality_vs_outcome TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.subagent_waste TO omninode_runtime;

-- Fail rather than ledger a partial grant when the runner cannot grant it.
SELECT 1 / (count(DISTINCT (c.relname, a.privilege_type)) = 15)::int
    AS work_questions_runtime_grants_assertion
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace,
     aclexplode(c.relacl) a
WHERE n.nspname = 'omninode_internal'
  AND c.relname IN (
      'work_outcome_units', 'orchestration_style_outcomes', 'lane_drift_and_stalls',
      'brief_quality_vs_outcome', 'subagent_waste'
  )
  AND a.grantee = 'omninode_runtime'::regrole
  AND a.privilege_type IN ('SELECT', 'INSERT', 'UPDATE');

COMMIT;
