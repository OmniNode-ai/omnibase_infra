-- OMN-14493 / OMN-14403 §4.1 (D4) + §12: the in-row outbox columns.
--
-- Adds two INFRA-OWNED denormalized columns to delegation_workflow_state so the
-- state_io stateful-dispatch wrapper can persist a leg's intended emissions IN
-- THE SAME optimistic-concurrency UPDATE that advances the FSM (commit-with-
-- intent), then publish-from-the-committed-row and CAS-finalize. This closes the
-- OMN-14493 CAS-retry result-selection race: a losing/retried CAS attempt can no
-- longer lose an already-emitted intent, because the winning row IS the publish
-- source (not the in-memory return of a possibly-losing attempt).
--
-- Why infra-owned columns, NOT the opaque `payload` JSONB (spec §4.1 D4/N5):
-- `payload` is codec-owned (omnimarket) and opaque to omnibase_infra by design.
-- Splicing an intended-emissions key into codec-owned JSON would be an unowned
-- cross-repo seam — exactly the field-match failure class OMN-14208 exists to
-- prevent. These columns sit alongside `tenant_id` / `state` / `in_flight`, the
-- existing infra-owned denormalized columns.
--
--   pending_emissions JSONB  — the batch of intended emissions. A list of entries;
--     each entry carries exactly what recovery needs to rebuild the emitted
--     envelope WITHOUT re-running the handler (spec §12): class_name, payload
--     (the event's own JSON), index, causation_envelope_id, correlation_id,
--     tenant_id. NULL / '[]'::jsonb = no un-published batch (the terminal/quiet
--     steady state).
--   publish_attempts INTEGER — row-level bounded-retry counter for the
--     poison-event → DLQ path (spec §9): a permanently-failing publish increments
--     this; after a bounded number it routes to DLQ + marks the row terminal-with-
--     error rather than redelivering forever.
--
-- Idempotent ADD COLUMN IF NOT EXISTS so warm dev/stability volumes reconcile
-- cleanly, matching migration 090's idempotent CREATE.

ALTER TABLE delegation_workflow_state
    ADD COLUMN IF NOT EXISTS pending_emissions JSONB,
    ADD COLUMN IF NOT EXISTS publish_attempts  INTEGER NOT NULL DEFAULT 0;

-- Recovery-select index: the boot/redeploy re-publish sweep (spec §4.1 D2/E3)
-- scans for in_flight rows carrying a non-empty pending_emissions batch,
-- REGARDLESS of TTL (a duplicate re-publish is benign — row-derived ids collapse
-- at the consume-path dedupe). Partial index keeps the scan cheap: only rows that
-- actually carry an un-published batch are indexed.
CREATE INDEX IF NOT EXISTS ix_delegation_workflow_state_recoverable_outbox
    ON delegation_workflow_state (updated_at)
    WHERE in_flight AND pending_emissions IS NOT NULL;
