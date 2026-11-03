-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-22311: remove default-privilege grants on the OMN-22320 tables.
--
-- 099's default privileges give omninode_runtime SELECT, INSERT and UPDATE
-- on every new omninode_internal table. The work-questions create migration
-- grants nothing, and the shipped topology declares no runtime access to its
-- five tables. Reconcile those accidental grants without changing the already
-- ledgered create migration or granting access to an undeclared consumer.
-- A future runtime consumer must declare and deliver its required privileges.
--
-- Scope: direct grants to omninode_runtime on these five tables only. Other
-- principals and tables are unchanged. As with the gateway-link-health revoke,
-- read the ACL after REVOKE: an executor without grant option can receive only
-- a warning. In that case the assertion aborts this transaction and the runner
-- must not ledger it. Reapplying after a successful revoke is a no-op.

BEGIN;

REVOKE ALL PRIVILEGES ON
    omninode_internal.work_outcome_units,
    omninode_internal.orchestration_style_outcomes,
    omninode_internal.lane_drift_and_stalls,
    omninode_internal.brief_quality_vs_outcome,
    omninode_internal.subagent_waste
FROM omninode_runtime;

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
