-- OMN-16770: the node's own record of which sessions it has already published.
-- Target DB: the APPLICATION database (omnidash_analytics on the compose lanes),
--            reached by the correlation pool via OMNINODE_INTERNAL_DB_URL.
-- Node: node_savings_estimation_compute
--
-- WHY THIS EXISTS
--   HandlerSavingsCorrelation._find_ready_sessions decides which sessions are
--   still un-finalized with an anti-join. Until this migration that anti-join
--   read `savings_estimates`:
--
--       AND NOT EXISTS (SELECT 1 FROM savings_estimates se WHERE ...)
--
--   `savings_estimates` is TENANT-domain. omnimarket's
--   node_projection_savings/081_savings_estimates_rls_tenant_isolation.sql puts
--   it under ENABLE + FORCE ROW LEVEL SECURITY with the policy
--   `tenant_id = current_setting('app.tenant_id', true)`, and this node neither
--   owns nor writes it. The correlation pool connects as `omninode_runtime`,
--   pinned NOSUPERUSER / NOBYPASSRLS / non-owner by OMN-16843.
--
--   Postgres row-level security fails OPEN from the caller's side: with the GUC
--   unset the policy evaluates to NULL for every row, the subquery matches
--   nothing, and `NOT EXISTS` becomes universally true -- every session reads as
--   never finalized and the batch re-publishes an estimate for every session on
--   every 60s tick. A bare GRANT SELECT is what PRODUCES that, not what fixes
--   it. OMN-16770's seam (_assert_idempotency_read_is_scoped) refuses the batch
--   rather than let it happen, which is correct, and permanent: this node
--   carries no tenant attribution of its own to bind (neither signal table in
--   0001 has a tenant_id column, and inventing one is what the OMN-16831 ruling
--   forbids). Measured on the .201 dev lane: 480 refusals in four hours, and no
--   estimate has ever been produced on any lane.
--
--   So the close named on OMN-16770 is to stop reading a TENANT relation for
--   INTERNAL idempotency at all. This relation is that close. The node writes
--   one row here per estimate it publishes and anti-joins THIS table instead --
--   a relation it owns, writes, and can read truthfully under its own binding.
--   The seam is unchanged and still runs immediately before the candidate
--   query; it now passes by construction rather than being satisfied by a
--   grant or an invented scope.
--
-- WHY THERE IS NO tenant_id COLUMN AND NO ROW LEVEL SECURITY HERE
--   Both would rebuild the defect. A GUC-predicated policy on the relation the
--   anti-join reads puts the batch straight back into the permanent refusal,
--   and a tenant_id this node has no attribution for would be invented
--   (OMN-16831). Finalization is a fact about THIS node's own publishing, not
--   about a tenant, so it is INTERNAL-domain. Ownership is declared in
--   omnimarket's scripts/application-relation-ownership.yaml (landed first, as
--   the OMN-15361 ownership gate requires).
--
-- WHY THERE IS NO OMN-15376 SHAPE-RECONCILIATION BLOCK
--   0001 carries one because its tables predate this migration tree and could
--   already exist with a drifted shape on a long-lived lane. This relation is
--   net-new and has never existed anywhere, so CREATE TABLE IF NOT EXISTS is
--   the whole story and a reconciliation block would be dead code.
--
-- ONE-TIME EFFECT ON A LANE THAT ALREADY HAS ESTIMATES
--   This table starts empty, so a session already present in `savings_estimates`
--   is re-published once. That is bounded, not unbounded: `candidate_sessions`
--   only considers sessions with a signal inside the lookback window (48h by
--   default), the re-published session is recorded here on that same tick, and
--   omnimarket's node_projection_savings upserts on session_id. It is not
--   backfilled from `savings_estimates` on purpose -- reading that relation to
--   seed this one is the exact cross-domain read being removed.
--
-- Idempotency: CREATE TABLE / INDEX guarded by IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS omninode_internal.savings_correlation_finalizations (
    session_id TEXT PRIMARY KEY,
    correlation_id UUID NOT NULL,
    finalized_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The anti-join probes by session_id, which the primary key already serves.
-- This index serves the operational question instead ("what did the batch
-- finalize in the last hour"), and the retention sweep that will eventually
-- trim rows older than the lookback window.
CREATE INDEX IF NOT EXISTS idx_savings_correlation_finalizations_finalized_at
    ON omninode_internal.savings_correlation_finalizations (finalized_at);

-- -----------------------------------------------------------------------------
-- omninode_runtime grant (topology-derived, same principal as 0001)
-- -----------------------------------------------------------------------------
-- SELECT for the anti-join, INSERT for the marker the batch writes after it
-- publishes. No UPDATE and no DELETE: a finalization is append-only, and the
-- batch has no reason to rewrite or retract one.
GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;
GRANT SELECT, INSERT ON omninode_internal.savings_correlation_finalizations TO omninode_runtime;
