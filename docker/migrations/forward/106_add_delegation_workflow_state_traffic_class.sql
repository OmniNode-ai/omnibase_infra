-- OMN-18172: make delegation traffic classification a typed, stored readback
-- surface without granting the chain canary access to the prompt-bearing JSONB
-- payload.
--
-- The authoritative value is the canonical provenance model persisted at
-- payload.request.provenance.traffic_class.  A STORED generated column keeps
-- the denormalized value transactionally identical to that source on INSERT
-- and UPDATE, including CAS updates made by the generic state_io adapter.
-- Legacy rows and callers that omit provenance remain explicitly
-- ``unclassified``; they are never inferred to be organic or synthetic from
-- prompt text.

ALTER TABLE delegation_workflow_state
    ADD COLUMN IF NOT EXISTS traffic_class TEXT
    GENERATED ALWAYS AS (
        COALESCE(
            payload #>> '{request,provenance,traffic_class}',
            'unclassified'
        )
    ) STORED;

ALTER TABLE delegation_workflow_state
    DROP CONSTRAINT IF EXISTS ck_delegation_workflow_state_traffic_class;
ALTER TABLE delegation_workflow_state
    ADD CONSTRAINT ck_delegation_workflow_state_traffic_class
    CHECK (traffic_class IN ('unclassified', 'organic', 'synthetic'));

COMMENT ON COLUMN delegation_workflow_state.traffic_class IS
    'OMN-18172 canonical delegation provenance class, stored from payload.request.provenance.traffic_class; omitted provenance is unclassified.';
