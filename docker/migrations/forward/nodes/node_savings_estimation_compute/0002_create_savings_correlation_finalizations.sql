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
-- WHY THERE IS AN OMN-15376 SHAPE-RECONCILIATION BLOCK ON A NET-NEW TABLE
--   A first draft omitted it, reasoning that the relation has never existed
--   anywhere so nothing can have drifted. tests/ci/
--   test_node_migration_shape_reconciliation.py rejected that, and it is right
--   to: "net-new" is a claim about the repository, not about any particular
--   lane. A lane that ran this file from a branch, an experiment, or a rolled
--   back deploy can already hold a table of this name with a different shape,
--   and CREATE TABLE IF NOT EXISTS SILENTLY NO-OPS against it -- after which
--   the first column-dependent statement raises `column "<col>" does not
--   exist` and ON_ERROR_STOP=1 kills the whole migration Job there. The
--   argument for skipping the block is exactly the argument that makes the
--   failure invisible until a lane deploy dies, so the block is here and the
--   gate is not exempted. On the fresh-create path every guarded add is a
--   no-op, so both paths end at the same schema. No DROP, no recreate, no
--   TRUNCATE.
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

-- ---- BEGIN OMN-15376 shape reconciliation: omninode_internal.savings_correlation_finalizations ----
-- Columns are added WITHOUT NOT NULL: on a drifted pre-existing table with
-- rows, ADD COLUMN ... NOT NULL with no default raises. The CREATE TABLE above
-- carries the full constraint set on the fresh-create path, which is the path
-- every lane is actually on; these guarded adds exist to keep a drifted lane
-- from dying at the first column-dependent statement, not to re-derive the
-- constraints.
ALTER TABLE omninode_internal.savings_correlation_finalizations ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE omninode_internal.savings_correlation_finalizations ADD COLUMN IF NOT EXISTS correlation_id UUID;
ALTER TABLE omninode_internal.savings_correlation_finalizations ADD COLUMN IF NOT EXISTS finalized_at TIMESTAMPTZ DEFAULT NOW();

-- The primary key is what makes the ON CONFLICT (session_id) in
-- HandlerSavingsCorrelation._record_finalization legal, so a drifted table
-- without it would fail every marker write rather than fail visibly here.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'omninode_internal.savings_correlation_finalizations'::regclass AND contype = 'p'
    ) THEN
        ALTER TABLE omninode_internal.savings_correlation_finalizations
            ADD CONSTRAINT savings_correlation_finalizations_pkey PRIMARY KEY (session_id);
    END IF;
END$$;

-- ---- END OMN-15376 shape reconciliation: omninode_internal.savings_correlation_finalizations ----

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
