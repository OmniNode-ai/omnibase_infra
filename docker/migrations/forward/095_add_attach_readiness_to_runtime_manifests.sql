-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration 095: carry boot attach-readiness on the runtime_manifests row (OMN-15512).
--
-- WHY:
--   The runtime already publishes what it WIRED (migration 079). What actually
--   ATTACHED was computed at boot into ModelRuntimeAttachReadiness and then
--   discarded into the log stream, so the only way to read the NOT-READY
--   blocker set was `ssh` + `docker logs omninode-runtime | grep NOT-READY`.
--   That is literally how OMN-15508 had to be diagnosed. These columns retire
--   that manual step: the blocker set with its named topics rides the SAME
--   event onto the SAME row — no new topic, no new reducer, no new table.
--
--   This matters even when /health is green: on 2026-07-30 the dev lane served
--   /health 200 healthy:true with registered_handlers ["db","http"] WHILE
--   NOT-READY warnings were still firing (34 in a trailing 5m window). Green
--   liveness is not evidence that consumers attached.
--
-- COLUMNS:
--   attach_state                — aggregate tri-state from EnumRuntimeReadinessState
--                                 ('ready' | 'degraded' | 'failed'), plus 'unknown'
--                                 for a boot where the per-contract interleave never
--                                 ran. 'unknown' is deliberately NOT 'ready': absence
--                                 of evidence is not evidence of attachment.
--   attach_required_contracts   — contracts the interleave walked.
--   attach_attached_contracts   — contracts whose consumer attached.
--   attach_not_ready_contracts  — the blocker set: one JSON object per contract that
--                                 did NOT attach, each a serialized
--                                 ModelContractAttachResult carrying contract_name,
--                                 status, topics_subscribed, the readiness confirm
--                                 outcome (including the failing topic names), and a
--                                 human-readable detail string. Bounded by the
--                                 not-attached count, NOT by the 475+ contracts walked
--                                 at boot: contracts that attached are already
--                                 enumerated in the `contracts`/`handlers` columns.
--
-- INVARIANT (holds whenever attach_state <> 'unknown'):
--   jsonb_array_length(attach_not_ready_contracts)
--     = attach_required_contracts - attach_attached_contracts
--
-- OPERATOR READBACK — replaces `docker logs | grep -c NOT-READY`:
--   SELECT started_at,
--          attach_state,
--          attach_attached_contracts || '/' || attach_required_contracts AS attached,
--          jsonb_array_length(attach_not_ready_contracts)                AS not_ready,
--          jsonb_agg(blocker -> 'contract_name')                         AS contracts,
--          jsonb_agg(blocker -> 'readiness' -> 'failures')               AS failing_topics
--     FROM runtime_manifests,
--          LATERAL jsonb_array_elements(attach_not_ready_contracts) AS blocker
--    WHERE runtime_profile = 'main'
--    GROUP BY started_at, attach_state, attach_attached_contracts,
--             attach_required_contracts, attach_not_ready_contracts
--    ORDER BY started_at DESC
--    LIMIT 1;
--
-- IDEMPOTENCY:
--   ADD COLUMN IF NOT EXISTS is safe to re-run.
--
-- BACKFILL:
--   None. runtime_manifests is append-only, one row per process boot; historical
--   rows predate the producer and correctly read attach_state='unknown'.
--
-- ROLLBACK:
--   ALTER TABLE runtime_manifests
--       DROP COLUMN IF EXISTS attach_state,
--       DROP COLUMN IF EXISTS attach_required_contracts,
--       DROP COLUMN IF EXISTS attach_attached_contracts,
--       DROP COLUMN IF EXISTS attach_not_ready_contracts;

ALTER TABLE runtime_manifests
    ADD COLUMN IF NOT EXISTS attach_state               TEXT   NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS attach_required_contracts  INT    NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS attach_attached_contracts  INT    NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS attach_not_ready_contracts JSONB  NOT NULL DEFAULT '[]';

-- Degraded-boot lookup: "which boots did not fully attach, most recent first".
-- Partial index — a fully-attached boot is the common case and is not indexed.
CREATE INDEX IF NOT EXISTS idx_runtime_manifests_attach_degraded
    ON runtime_manifests (runtime_profile, started_at DESC)
    WHERE attach_state <> 'ready';

COMMENT ON COLUMN runtime_manifests.attach_state IS
    'Aggregate boot attach tri-state (OMN-15512): ready | degraded | failed, or '
    'unknown when the per-contract interleave did not run. Never infer '
    'attachment from process liveness — a green /health can coexist with '
    'attach_state=degraded.';

COMMENT ON COLUMN runtime_manifests.attach_not_ready_contracts IS
    'Blocker set (OMN-15512): serialized ModelContractAttachResult per contract '
    'that did NOT attach, including the topics whose readiness confirm failed. '
    'Replaces `docker logs omninode-runtime | grep NOT-READY`.';
