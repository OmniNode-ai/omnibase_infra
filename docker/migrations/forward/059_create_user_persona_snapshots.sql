-- Migration 058: Create user_persona_snapshots table
-- Phase 3: Adaptive Personalization (OMN-7305, OMN-3970)
--
-- Append-only persona snapshot storage. Each persona rebuild creates a new
-- versioned row. No updates or deletes.
--
-- Consent enforcement deferred to Phase B (OMN-3980).

CREATE TABLE IF NOT EXISTS user_persona_snapshots (
    persona_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id              VARCHAR(128) NOT NULL,
    agent_id             VARCHAR(64),
    technical_level      VARCHAR(32) NOT NULL DEFAULT 'intermediate',
    vocabulary_complexity DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    preferred_tone       VARCHAR(32) NOT NULL DEFAULT 'explanatory',
    domain_familiarity   JSONB NOT NULL DEFAULT '{}',
    session_count        INTEGER NOT NULL DEFAULT 0,
    persona_version      INTEGER NOT NULL DEFAULT 1,
    rebuilt_from_signals INTEGER NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_persona_user_version
    ON user_persona_snapshots (user_id, persona_version DESC);

CREATE INDEX IF NOT EXISTS idx_persona_agent
    ON user_persona_snapshots (agent_id) WHERE agent_id IS NOT NULL;
