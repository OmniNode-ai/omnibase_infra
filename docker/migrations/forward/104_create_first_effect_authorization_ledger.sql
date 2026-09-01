-- Durable, value-redacted observation ledger for a workflow's first effect.
--
-- This relation contains only opaque SHA-256 digests, UUID identity, state,
-- timestamps, and an optimistic version. It never stores an event body, a
-- secret, request payload, model prompt, or model output.
--
-- It is an observation/recording scaffold, not a publish mechanism. The canonical outbox is the
-- state-IO in-row outbox (migration 093), which owns producer payloads and its
-- publisher. This migration deliberately has no payload or outbox binding, so
-- its PUBLISHING state is an observation only and confers no effect permission.
-- The authorization-named relation and identifiers are factual schema names,
-- not grants. Production publishing must fail closed unless an external
-- transactional canonical-outbox stager owns both the intent and its ledger
-- transition.
--
-- This is a one-time forward migration, not a reconciliation migration. An
-- existing relation named public.first_effect_authorization_ledger is rejected
-- by CREATE TABLE rather than silently accepted: a pre-existing object has no
-- verified compatible shape, ownership, constraints, or trigger.
--
-- No GRANT or ALTER OWNER appears here. No runtime caller is wired by this
-- payload-free observation scaffold; it creates no effect permission.
--
-- Lifecycle observations:
-- ISSUED -> PREFLIGHT_CONSUMED -> PUBLISHING
-- PUBLISHING -> PUBLISHED_UNKNOWN | TERMINAL_OBSERVED | BLOCKED
-- PUBLISHED_UNKNOWN -> TERMINAL_OBSERVED | BLOCKED

CREATE TABLE public.first_effect_authorization_ledger (
    authorization_digest      TEXT PRIMARY KEY,
    nonce_digest              TEXT NOT NULL UNIQUE,
    correlation_id            UUID NOT NULL UNIQUE,
    request_digest            TEXT NOT NULL UNIQUE,
    manifest_hash             TEXT NOT NULL,
    state                     TEXT NOT NULL,
    publish_evidence_hash     TEXT,
    terminal_receipt_hash     TEXT,
    issued_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    preflight_consumed_at     TIMESTAMPTZ,
    publishing_at             TIMESTAMPTZ,
    published_unknown_at      TIMESTAMPTZ,
    terminal_observed_at      TIMESTAMPTZ,
    blocked_at                TIMESTAMPTZ,
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version                   BIGINT NOT NULL DEFAULT 0,
    CONSTRAINT ck_first_effect_authorization_digest
        CHECK (authorization_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_first_effect_nonce_digest
        CHECK (nonce_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_first_effect_request_digest
        CHECK (request_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_first_effect_manifest_hash
        CHECK (manifest_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_first_effect_publish_evidence_hash
        CHECK (publish_evidence_hash IS NULL OR publish_evidence_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_first_effect_terminal_receipt_hash
        CHECK (terminal_receipt_hash IS NULL OR terminal_receipt_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_first_effect_state
        CHECK (state IN (
            'ISSUED', 'PREFLIGHT_CONSUMED', 'PUBLISHING', 'PUBLISHED_UNKNOWN',
            'TERMINAL_OBSERVED', 'BLOCKED'
        )),
    CONSTRAINT ck_first_effect_version_nonnegative CHECK (version >= 0),
    CONSTRAINT ck_first_effect_state_evidence_shape CHECK (
        (state = 'ISSUED'
            AND preflight_consumed_at IS NULL AND publishing_at IS NULL
            AND published_unknown_at IS NULL AND terminal_observed_at IS NULL
            AND blocked_at IS NULL AND publish_evidence_hash IS NULL
            AND terminal_receipt_hash IS NULL)
        OR (state = 'PREFLIGHT_CONSUMED'
            AND preflight_consumed_at IS NOT NULL AND publishing_at IS NULL
            AND published_unknown_at IS NULL AND terminal_observed_at IS NULL
            AND blocked_at IS NULL AND publish_evidence_hash IS NULL
            AND terminal_receipt_hash IS NULL)
        OR (state = 'PUBLISHING'
            AND preflight_consumed_at IS NOT NULL AND publishing_at IS NOT NULL
            AND published_unknown_at IS NULL AND terminal_observed_at IS NULL
            AND blocked_at IS NULL AND publish_evidence_hash IS NULL
            AND terminal_receipt_hash IS NULL)
        OR (state = 'PUBLISHED_UNKNOWN'
            AND preflight_consumed_at IS NOT NULL AND publishing_at IS NOT NULL
            AND published_unknown_at IS NOT NULL AND terminal_observed_at IS NULL
            AND blocked_at IS NULL AND publish_evidence_hash IS NOT NULL
            AND terminal_receipt_hash IS NULL)
        OR (state = 'TERMINAL_OBSERVED'
            AND preflight_consumed_at IS NOT NULL AND publishing_at IS NOT NULL
            AND terminal_observed_at IS NOT NULL AND blocked_at IS NULL
            AND publish_evidence_hash IS NOT NULL AND terminal_receipt_hash IS NOT NULL)
        OR (state = 'BLOCKED'
            AND preflight_consumed_at IS NOT NULL AND publishing_at IS NOT NULL
            AND blocked_at IS NOT NULL AND terminal_observed_at IS NULL
            AND terminal_receipt_hash IS NULL)
    )
);

CREATE INDEX ix_first_effect_authorization_reobservation
    ON public.first_effect_authorization_ledger (updated_at)
    WHERE state = 'PUBLISHED_UNKNOWN';

CREATE FUNCTION public.enforce_first_effect_authorization_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $function$
BEGIN
    IF NEW.authorization_digest <> OLD.authorization_digest
       OR NEW.nonce_digest <> OLD.nonce_digest
       OR NEW.correlation_id <> OLD.correlation_id
       OR NEW.request_digest <> OLD.request_digest
       OR NEW.manifest_hash <> OLD.manifest_hash
       OR NEW.issued_at <> OLD.issued_at THEN
        RAISE EXCEPTION 'first-effect authorization identity is immutable';
    END IF;

    IF (OLD.preflight_consumed_at IS NOT NULL AND NEW.preflight_consumed_at IS DISTINCT FROM OLD.preflight_consumed_at)
       OR (OLD.publishing_at IS NOT NULL AND NEW.publishing_at IS DISTINCT FROM OLD.publishing_at)
       OR (OLD.published_unknown_at IS NOT NULL AND NEW.published_unknown_at IS DISTINCT FROM OLD.published_unknown_at)
       OR (OLD.terminal_observed_at IS NOT NULL AND NEW.terminal_observed_at IS DISTINCT FROM OLD.terminal_observed_at)
       OR (OLD.blocked_at IS NOT NULL AND NEW.blocked_at IS DISTINCT FROM OLD.blocked_at)
       OR (OLD.publish_evidence_hash IS NOT NULL AND NEW.publish_evidence_hash IS DISTINCT FROM OLD.publish_evidence_hash)
       OR (OLD.terminal_receipt_hash IS NOT NULL AND NEW.terminal_receipt_hash IS DISTINCT FROM OLD.terminal_receipt_hash) THEN
        RAISE EXCEPTION 'first-effect authorization evidence and transition timestamps are immutable';
    END IF;

    IF NEW.version <> OLD.version + 1 THEN
        RAISE EXCEPTION 'first-effect authorization version must advance by one';
    END IF;

    IF (OLD.state = 'ISSUED' AND NEW.state = 'PREFLIGHT_CONSUMED')
       OR (OLD.state = 'PREFLIGHT_CONSUMED' AND NEW.state = 'PUBLISHING')
       OR (OLD.state = 'PUBLISHING' AND NEW.state IN ('PUBLISHED_UNKNOWN', 'TERMINAL_OBSERVED', 'BLOCKED'))
       OR (OLD.state = 'PUBLISHED_UNKNOWN' AND NEW.state IN ('TERMINAL_OBSERVED', 'BLOCKED')) THEN
        RETURN NEW;
    END IF;

    RAISE EXCEPTION 'illegal first-effect authorization transition: % -> %', OLD.state, NEW.state;
END;
$function$;

CREATE TRIGGER trg_first_effect_authorization_transition
    BEFORE UPDATE ON public.first_effect_authorization_ledger
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_first_effect_authorization_transition();
