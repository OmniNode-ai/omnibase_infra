-- OMN-16293: raw signal-capture tables for savings-estimation correlation.
-- Target DB: omnibase_infra primary Postgres (.201:5436).
-- Node: node_savings_estimation_compute
--
-- WHY THESE EXIST
--   node_savings_estimation_compute's periodic correlation batch
--   (HandlerSavingsCorrelation.run_correlation_batch, OMN-16293) needs
--   per-session injection-effectiveness and validator-catch signals to build
--   a ModelSavingsEstimationInput. Unlike llm-call-completed and
--   session-outcome (already projected by omnimarket's node_projection_llm_cost
--   / node_projection_session_outcome into llm_call_metrics / session_outcomes),
--   NO existing projection captures onex.evt.omniclaude.context-injected.v1,
--   onex.evt.omniclaude.validator-catch.v1, or
--   onex.evt.omniclaude.pattern-enforcement.v1 anywhere in the platform —
--   confirmed by a repo-wide grep across omnibase_infra and omnimarket
--   migrations turning up zero hits before this file was authored.
--
--   Design note (OMN-16293 architecture decision): every raw event is
--   INSERTed immediately on ingest -- there is no in-memory correlation
--   buffer. Session-level correlation state lives entirely in these tables
--   (the "projection surface"), read fresh by each periodic batch tick. This
--   supersedes the legacy ServiceSavingsEstimator in-memory
--   SessionBuffer/OrderedDict design (services/observability/savings_estimation/
--   consumer.py, deleted in this PR).
--
-- Idempotency: CREATE TABLE / INDEX guarded by IF NOT EXISTS.

-- ============================================================================
-- SAVINGS_INJECTION_SIGNALS TABLE
-- ============================================================================
-- One row per onex.evt.omniclaude.context-injected.v1 event. Append-only.
CREATE TABLE IF NOT EXISTS omninode_internal.savings_injection_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    tokens_injected INTEGER NOT NULL CHECK (tokens_injected >= 0),
    patterns_count INTEGER NOT NULL DEFAULT 0 CHECK (patterns_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---- BEGIN OMN-15376 shape reconciliation: omninode_internal.savings_injection_signals ----
-- The CREATE TABLE IF NOT EXISTS above SILENTLY NO-OPS when a table of this
-- name already exists with a DIFFERENT shape. Everything below it in this
-- file is NOT so forgiving: CREATE INDEX IF NOT EXISTS guards the index
-- NAME, not the COLUMN, so the first column-dependent statement raises
--   ERROR: column "<col>" does not exist
-- and ON_ERROR_STOP=1 kills the whole migration Job there (OMN-15376 class).
--
-- The guarded adds below converge a drifted pre-existing table onto the
-- shape declared above. On the fresh-create path every one is a no-op (the
-- column already exists), so BOTH paths end at the same schema. No DROP, no
-- recreate, no TRUNCATE: pre-existing rows are preserved.

ALTER TABLE omninode_internal.savings_injection_signals ADD COLUMN IF NOT EXISTS id UUID DEFAULT gen_random_uuid();
ALTER TABLE omninode_internal.savings_injection_signals ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE omninode_internal.savings_injection_signals ADD COLUMN IF NOT EXISTS tokens_injected INTEGER;
ALTER TABLE omninode_internal.savings_injection_signals ADD COLUMN IF NOT EXISTS patterns_count INTEGER DEFAULT 0;
ALTER TABLE omninode_internal.savings_injection_signals ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

DO $$
DECLARE
    v_col  TEXT;
    v_nulls BIGINT;
BEGIN
    FOREACH v_col IN ARRAY ARRAY['id', 'session_id', 'tokens_injected', 'patterns_count', 'created_at']
    LOOP
        EXECUTE format(
            'SELECT count(*) FROM %s WHERE %I IS NULL', 'omninode_internal.savings_injection_signals'::regclass, v_col
        ) INTO v_nulls;
        IF v_nulls = 0 THEN
            EXECUTE format(
                'ALTER TABLE %s ALTER COLUMN %I SET NOT NULL', 'omninode_internal.savings_injection_signals'::regclass, v_col
            );
        ELSE
            RAISE EXCEPTION
                'OMN-15376: cannot converge omninode_internal.savings_injection_signals.% to NOT NULL -- % pre-existing row(s) hold NULL. This needs a data ruling (backfill value, or drop the NOT NULL from the contract); the migration refuses to guess.',
                v_col, v_nulls;
        END IF;
    END LOOP;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'omninode_internal.savings_injection_signals'::regclass AND contype = 'p'
    ) THEN
        ALTER TABLE omninode_internal.savings_injection_signals ADD CONSTRAINT savings_injection_signals_pkey PRIMARY KEY (id);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'omninode_internal.savings_injection_signals'::regclass AND conname = 'savings_injection_signals_tokens_injected_check'
    ) THEN
        ALTER TABLE omninode_internal.savings_injection_signals ADD CONSTRAINT savings_injection_signals_tokens_injected_check CHECK (tokens_injected >= 0);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'omninode_internal.savings_injection_signals'::regclass AND conname = 'savings_injection_signals_patterns_count_check'
    ) THEN
        ALTER TABLE omninode_internal.savings_injection_signals ADD CONSTRAINT savings_injection_signals_patterns_count_check CHECK (patterns_count >= 0);
    END IF;
END$$;

-- ---- END OMN-15376 shape reconciliation: omninode_internal.savings_injection_signals ----

CREATE INDEX IF NOT EXISTS idx_savings_injection_signals_session_id
    ON omninode_internal.savings_injection_signals (session_id);

CREATE INDEX IF NOT EXISTS idx_savings_injection_signals_created_at
    ON omninode_internal.savings_injection_signals (created_at);

-- ============================================================================
-- SAVINGS_VALIDATOR_CATCH_SIGNALS TABLE
-- ============================================================================
-- One row per onex.evt.omniclaude.validator-catch.v1 or
-- onex.evt.omniclaude.pattern-enforcement.v1 event. Append-only.
CREATE TABLE IF NOT EXISTS omninode_internal.savings_validator_catch_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    validator_type TEXT NOT NULL DEFAULT '',
    source_event_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT savings_validator_catch_signals_severity_check
        CHECK (severity IN ('critical', 'major', 'minor')),
    CONSTRAINT savings_validator_catch_signals_source_event_type_check
        CHECK (source_event_type IN ('validator-catch', 'pattern-enforcement'))
);

-- ---- BEGIN OMN-15376 shape reconciliation: omninode_internal.savings_validator_catch_signals ----
ALTER TABLE omninode_internal.savings_validator_catch_signals ADD COLUMN IF NOT EXISTS id UUID DEFAULT gen_random_uuid();
ALTER TABLE omninode_internal.savings_validator_catch_signals ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE omninode_internal.savings_validator_catch_signals ADD COLUMN IF NOT EXISTS severity TEXT;
ALTER TABLE omninode_internal.savings_validator_catch_signals ADD COLUMN IF NOT EXISTS validator_type TEXT DEFAULT '';
ALTER TABLE omninode_internal.savings_validator_catch_signals ADD COLUMN IF NOT EXISTS source_event_type TEXT;
ALTER TABLE omninode_internal.savings_validator_catch_signals ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

DO $$
DECLARE
    v_col  TEXT;
    v_nulls BIGINT;
BEGIN
    FOREACH v_col IN ARRAY ARRAY['id', 'session_id', 'severity', 'validator_type', 'source_event_type', 'created_at']
    LOOP
        EXECUTE format(
            'SELECT count(*) FROM %s WHERE %I IS NULL', 'omninode_internal.savings_validator_catch_signals'::regclass, v_col
        ) INTO v_nulls;
        IF v_nulls = 0 THEN
            EXECUTE format(
                'ALTER TABLE %s ALTER COLUMN %I SET NOT NULL', 'omninode_internal.savings_validator_catch_signals'::regclass, v_col
            );
        ELSE
            RAISE EXCEPTION
                'OMN-15376: cannot converge omninode_internal.savings_validator_catch_signals.% to NOT NULL -- % pre-existing row(s) hold NULL. This needs a data ruling (backfill value, or drop the NOT NULL from the contract); the migration refuses to guess.',
                v_col, v_nulls;
        END IF;
    END LOOP;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'omninode_internal.savings_validator_catch_signals'::regclass AND contype = 'p'
    ) THEN
        ALTER TABLE omninode_internal.savings_validator_catch_signals ADD CONSTRAINT savings_validator_catch_signals_pkey PRIMARY KEY (id);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'omninode_internal.savings_validator_catch_signals'::regclass AND conname = 'savings_validator_catch_signals_severity_check'
    ) THEN
        ALTER TABLE omninode_internal.savings_validator_catch_signals ADD CONSTRAINT savings_validator_catch_signals_severity_check CHECK (severity IN ('critical', 'major', 'minor'));
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'omninode_internal.savings_validator_catch_signals'::regclass AND conname = 'savings_validator_catch_signals_source_event_type_check'
    ) THEN
        ALTER TABLE omninode_internal.savings_validator_catch_signals ADD CONSTRAINT savings_validator_catch_signals_source_event_type_check CHECK (source_event_type IN ('validator-catch', 'pattern-enforcement'));
    END IF;
END$$;

-- ---- END OMN-15376 shape reconciliation: omninode_internal.savings_validator_catch_signals ----

CREATE INDEX IF NOT EXISTS idx_savings_validator_catch_signals_session_id
    ON omninode_internal.savings_validator_catch_signals (session_id);

CREATE INDEX IF NOT EXISTS idx_savings_validator_catch_signals_created_at
    ON omninode_internal.savings_validator_catch_signals (created_at);

-- -----------------------------------------------------------------------------
-- omninode_runtime grant (topology-derived, OMN-16293)
-- -----------------------------------------------------------------------------
GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.savings_injection_signals TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.savings_validator_catch_signals TO omninode_runtime;
