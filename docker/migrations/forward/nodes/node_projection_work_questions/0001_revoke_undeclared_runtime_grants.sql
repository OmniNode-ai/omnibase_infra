-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-21698: keep fresh runtime builds within the declared topology ACLs.
-- 099's schema-wide default privileges grant SELECT, INSERT and UPDATE on
-- these tables when 0000 creates them. The topology declares no runtime
-- consumer of these five relations; the work-question writer is a separate
-- omnimarket change. A future consumer must declare and deliver its grants
-- together rather than inherit privileges from the schema-wide default.
-- This successor also repairs databases that already applied 0000, whose
-- checksum-bound bytes must remain unchanged.

BEGIN;

REVOKE ALL PRIVILEGES ON
    omninode_internal.work_outcome_units,
    omninode_internal.orchestration_style_outcomes,
    omninode_internal.lane_drift_and_stalls,
    omninode_internal.brief_quality_vs_outcome,
    omninode_internal.subagent_waste
FROM omninode_runtime;

-- A non-grantor's REVOKE may only warn. Read the relation ACLs directly so
-- that case fails the migration, rolls back, and cannot enter the ledger.
SELECT 1 / (count(*) = 0)::int AS work_questions_runtime_revoked_assertion
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace,
     aclexplode(c.relacl) a
WHERE n.nspname = 'omninode_internal'
  AND c.relname IN (
      'work_outcome_units',
      'orchestration_style_outcomes',
      'lane_drift_and_stalls',
      'brief_quality_vs_outcome',
      'subagent_waste'
  )
  AND a.grantee = 'omninode_runtime'::regrole;

COMMIT;
