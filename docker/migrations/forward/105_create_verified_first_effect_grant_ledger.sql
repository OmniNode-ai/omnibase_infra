-- Durable first-effect authority boundary.
--
-- This relation is populated only after an authenticated verifier has accepted
-- a signed activation grant.  It records causal projections and state, never a
-- signed blob, key, request body, event body, prompt, or model output.  The
-- workflow row's pending_emissions column remains the single payload owner.
--
-- This migration intentionally introduces a new relation instead of upgrading
-- migration 104's observation scaffold: an old observation cannot truthfully be
-- reclassified as a verified grant.  There is no publisher or broker operation
-- in this schema.

CREATE TABLE public.first_effect_verified_grant_ledger (
    authorization_digest              TEXT PRIMARY KEY,
    grant_id                          UUID NOT NULL UNIQUE,
    grant_envelope_id                 UUID NOT NULL UNIQUE,
    nonce_digest                      TEXT NOT NULL UNIQUE,
    request_digest                    TEXT NOT NULL UNIQUE,
    correlation_id                    TEXT NOT NULL UNIQUE
                                        REFERENCES public.delegation_workflow_state(correlation_id),
    tenant_id                         TEXT NOT NULL,
    backend_id                        TEXT NOT NULL,
    rendered_contract_sha256          TEXT NOT NULL,
    issuer_key_fingerprint_sha256     TEXT NOT NULL,
    retry_disposition                 TEXT NOT NULL,
    expected_output_topic             TEXT NOT NULL,
    expected_output_event_class       TEXT NOT NULL,
    expected_output_event_index       INTEGER NOT NULL,
    state                             TEXT NOT NULL,
    outbox_envelope_id                UUID UNIQUE,
    outbox_body_sha256                TEXT,
    outbox_topic                      TEXT,
    outbox_event_class                TEXT,
    outbox_event_index                INTEGER,
    workflow_version                  INTEGER,
    verified_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    staged_at                         TIMESTAMPTZ,
    publishing_at                     TIMESTAMPTZ,
    claimed_at                        TIMESTAMPTZ,
    published_unknown_at              TIMESTAMPTZ,
    terminal_at                       TIMESTAMPTZ,
    updated_at                        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version                           BIGINT NOT NULL DEFAULT 0,
    CONSTRAINT ck_verified_first_effect_authorization_digest
        CHECK (authorization_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_verified_first_effect_nonce_digest
        CHECK (nonce_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_verified_first_effect_request_digest
        CHECK (request_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_verified_first_effect_rendered_contract_digest
        CHECK (rendered_contract_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_verified_first_effect_issuer_fingerprint
        CHECK (issuer_key_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_verified_first_effect_outbox_body_digest
        CHECK (outbox_body_sha256 IS NULL OR outbox_body_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_verified_first_effect_retry_disposition
        CHECK (retry_disposition = 'never-republish-after-ambiguous.v1'),
    CONSTRAINT ck_verified_first_effect_expected_topic
        CHECK (expected_output_topic ~ '^onex\.[a-z0-9-]+(\.[a-z0-9-]+)+\.v[1-9][0-9]*$'),
    CONSTRAINT ck_verified_first_effect_expected_event_class
        CHECK (expected_output_event_class ~ '^Model[A-Za-z0-9]+$'),
    CONSTRAINT ck_verified_first_effect_expected_event_index
        CHECK (expected_output_event_index = 0),
    CONSTRAINT ck_verified_first_effect_state
        CHECK (state IN ('VERIFIED', 'STAGED', 'PUBLISHING', 'CLAIMED', 'PUBLISHED_UNKNOWN', 'TERMINAL')),
    CONSTRAINT ck_verified_first_effect_version_nonnegative CHECK (version >= 0),
    CONSTRAINT ck_verified_first_effect_shape CHECK (
        (state = 'VERIFIED'
            AND outbox_envelope_id IS NULL AND outbox_body_sha256 IS NULL
            AND outbox_topic IS NULL AND outbox_event_class IS NULL
            AND outbox_event_index IS NULL AND workflow_version IS NULL
            AND staged_at IS NULL AND publishing_at IS NULL AND claimed_at IS NULL
            AND published_unknown_at IS NULL AND terminal_at IS NULL)
        OR (state = 'STAGED'
            AND outbox_envelope_id IS NOT NULL AND outbox_body_sha256 IS NOT NULL
            AND outbox_topic = expected_output_topic
            AND outbox_event_class = expected_output_event_class
            AND outbox_event_index = expected_output_event_index
            AND workflow_version IS NOT NULL AND staged_at IS NOT NULL
            AND publishing_at IS NULL AND claimed_at IS NULL
            AND published_unknown_at IS NULL AND terminal_at IS NULL)
        OR (state = 'PUBLISHING'
            AND outbox_envelope_id IS NOT NULL AND outbox_body_sha256 IS NOT NULL
            AND outbox_topic = expected_output_topic
            AND outbox_event_class = expected_output_event_class
            AND outbox_event_index = expected_output_event_index
            AND workflow_version IS NOT NULL AND staged_at IS NOT NULL
            AND publishing_at IS NOT NULL AND claimed_at IS NULL
            AND published_unknown_at IS NULL AND terminal_at IS NULL)
        OR (state = 'PUBLISHED_UNKNOWN'
            AND outbox_envelope_id IS NOT NULL AND outbox_body_sha256 IS NOT NULL
            AND outbox_topic = expected_output_topic
            AND outbox_event_class = expected_output_event_class
            AND outbox_event_index = expected_output_event_index
            AND workflow_version IS NOT NULL AND staged_at IS NOT NULL
            AND publishing_at IS NOT NULL AND claimed_at IS NULL
            AND published_unknown_at IS NOT NULL AND terminal_at IS NULL)
        OR (state = 'CLAIMED'
            AND outbox_envelope_id IS NOT NULL AND outbox_body_sha256 IS NOT NULL
            AND outbox_topic = expected_output_topic
            AND outbox_event_class = expected_output_event_class
            AND outbox_event_index = expected_output_event_index
            AND workflow_version IS NOT NULL AND staged_at IS NOT NULL
            AND publishing_at IS NOT NULL AND claimed_at IS NOT NULL AND terminal_at IS NULL)
        OR (state = 'TERMINAL'
            AND outbox_envelope_id IS NOT NULL AND outbox_body_sha256 IS NOT NULL
            AND outbox_topic = expected_output_topic
            AND outbox_event_class = expected_output_event_class
            AND outbox_event_index = expected_output_event_index
            AND workflow_version IS NOT NULL AND staged_at IS NOT NULL
            AND publishing_at IS NOT NULL AND claimed_at IS NOT NULL AND terminal_at IS NOT NULL)
    ),
    CONSTRAINT uq_verified_first_effect_output_identity
        UNIQUE (outbox_body_sha256, outbox_topic, outbox_event_class, outbox_event_index)
);

CREATE FUNCTION public.enforce_verified_first_effect_grant_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $function$
BEGIN
    IF NEW.authorization_digest <> OLD.authorization_digest
       OR NEW.grant_id <> OLD.grant_id
       OR NEW.grant_envelope_id <> OLD.grant_envelope_id
       OR NEW.nonce_digest <> OLD.nonce_digest
       OR NEW.request_digest <> OLD.request_digest
       OR NEW.correlation_id <> OLD.correlation_id
       OR NEW.tenant_id <> OLD.tenant_id
       OR NEW.backend_id <> OLD.backend_id
       OR NEW.rendered_contract_sha256 <> OLD.rendered_contract_sha256
       OR NEW.issuer_key_fingerprint_sha256 <> OLD.issuer_key_fingerprint_sha256
       OR NEW.retry_disposition <> OLD.retry_disposition
       OR NEW.expected_output_topic <> OLD.expected_output_topic
       OR NEW.expected_output_event_class <> OLD.expected_output_event_class
       OR NEW.expected_output_event_index <> OLD.expected_output_event_index
       OR NEW.verified_at <> OLD.verified_at THEN
        RAISE EXCEPTION 'verified first-effect grant causal projection is immutable';
    END IF;

    IF (OLD.outbox_envelope_id IS NOT NULL AND NEW.outbox_envelope_id IS DISTINCT FROM OLD.outbox_envelope_id)
       OR (OLD.outbox_body_sha256 IS NOT NULL AND NEW.outbox_body_sha256 IS DISTINCT FROM OLD.outbox_body_sha256)
       OR (OLD.outbox_topic IS NOT NULL AND NEW.outbox_topic IS DISTINCT FROM OLD.outbox_topic)
       OR (OLD.outbox_event_class IS NOT NULL AND NEW.outbox_event_class IS DISTINCT FROM OLD.outbox_event_class)
       OR (OLD.outbox_event_index IS NOT NULL AND NEW.outbox_event_index IS DISTINCT FROM OLD.outbox_event_index)
       OR (OLD.workflow_version IS NOT NULL AND NEW.workflow_version IS DISTINCT FROM OLD.workflow_version)
       OR (OLD.staged_at IS NOT NULL AND NEW.staged_at IS DISTINCT FROM OLD.staged_at)
       OR (OLD.publishing_at IS NOT NULL AND NEW.publishing_at IS DISTINCT FROM OLD.publishing_at)
       OR (OLD.claimed_at IS NOT NULL AND NEW.claimed_at IS DISTINCT FROM OLD.claimed_at)
       OR (OLD.published_unknown_at IS NOT NULL AND NEW.published_unknown_at IS DISTINCT FROM OLD.published_unknown_at)
       OR (OLD.terminal_at IS NOT NULL AND NEW.terminal_at IS DISTINCT FROM OLD.terminal_at) THEN
        RAISE EXCEPTION 'verified first-effect grant staged binding is immutable';
    END IF;

    IF NEW.version <> OLD.version + 1 THEN
        RAISE EXCEPTION 'verified first-effect grant version must advance by one';
    END IF;

    IF (OLD.state = 'VERIFIED' AND NEW.state = 'STAGED')
       OR (OLD.state = 'STAGED' AND NEW.state = 'PUBLISHING')
       OR (OLD.state = 'PUBLISHING' AND NEW.state IN ('PUBLISHED_UNKNOWN', 'CLAIMED'))
       OR (OLD.state = 'PUBLISHED_UNKNOWN' AND NEW.state = 'CLAIMED')
       OR (OLD.state = 'CLAIMED' AND NEW.state = 'TERMINAL') THEN
        RETURN NEW;
    END IF;

    RAISE EXCEPTION 'illegal verified first-effect grant transition: % -> %', OLD.state, NEW.state;
END;
$function$;

CREATE TRIGGER trg_verified_first_effect_grant_transition
    BEFORE UPDATE ON public.first_effect_verified_grant_ledger
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_verified_first_effect_grant_transition();
