-- OMN-19561 AC3: duplicate/stale event absorbs must not reset the completion
-- bound used by StateStoreAdapter.select_abandoned_rows. Only a change to
-- state, payload, or tenant_id refreshes updated_at; CAS version bumps and
-- outbox bookkeeping alone preserve the last transition timestamp.
-- Replaces migration 090's function in place, retaining its existing trigger.

CREATE OR REPLACE FUNCTION refresh_delegation_workflow_state_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.state IS NOT DISTINCT FROM OLD.state
       AND NEW.payload IS NOT DISTINCT FROM OLD.payload
       AND NEW.tenant_id IS NOT DISTINCT FROM OLD.tenant_id THEN
        NEW.updated_at = OLD.updated_at;
    ELSE
        NEW.updated_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
