-- OMN-14576: single-publisher claim for the in-row outbox boot-recovery sweep.
--
-- The OMN-14493 outbox (migration 093) persists a leg's intended emissions in
-- pending_emissions and re-publishes them from the row on boot recovery. That
-- recovery is gated per-PROCESS (_recovery_ran), so two processes racing boot
-- recovery — a rolling redeploy overlap or a multi-pod deploy — could EACH
-- re-publish the same recovered batch (a cross-process double-publish).
--
-- These two columns let the recovery sweep CAS-CLAIM a row before publishing it,
-- using the existing optimistic-concurrency machinery (a version-gated UPDATE) —
-- NOT a distributed lock. Only the instance whose CAS wins publishes a given
-- batch; a concurrent sweep's claim fails the version check (or the fresh-claim
-- guard below) and skips.
--
--   recovery_claimed_by TEXT        — the instance-id / claim token of the sweep
--     that owns the in-flight re-publish. NULL = unclaimed (recoverable).
--   recovery_claimed_at TIMESTAMPTZ — when the claim was taken. A claim is only
--     honored (excluded from re-selection) while FRESH — within
--     DELEGATION_RECOVERY_CLAIM_TTL_SECONDS (default 120s, << the 900s stale-FAIL
--     TTL). A claimer that crashes mid-publish leaves a stale claim; once it ages
--     past the claim-TTL the row is re-selectable and a later sweep re-claims +
--     re-publishes (at-least-once — the residual caught by the idempotent-consumer
--     end-state, OMN-14577). Distinct from migration 093's stale-FAIL give-up TTL,
--     so a genuinely-crashed row (recovery_claimed_at IS NULL) is still recovered
--     PROMPTLY at boot regardless of TTL (spec §4.1 E3).
--
-- Idempotent ADD COLUMN IF NOT EXISTS so warm dev/stability volumes reconcile.

ALTER TABLE delegation_workflow_state
    ADD COLUMN IF NOT EXISTS recovery_claimed_by TEXT,
    ADD COLUMN IF NOT EXISTS recovery_claimed_at  TIMESTAMPTZ;
