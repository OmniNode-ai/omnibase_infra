-- Migration: 068_create_landing_content_projection
-- Description: Projection table for onex.evt.omniweb.content-updated.v1 events.
-- Consumed by node_content_projection (omnimarket). UPSERT key: (content_kind, schema_version_major).
-- Idempotency: All statements use IF NOT EXISTS; safe to re-apply.

CREATE TABLE IF NOT EXISTS landing_content_projection (
    content_kind            VARCHAR(128)    NOT NULL,
    schema_version_major    INT             NOT NULL DEFAULT 1,
    data                    JSONB           NOT NULL,
    last_applied_event_id   UUID            NOT NULL,
    last_applied_offset     BIGINT          NOT NULL DEFAULT 0,
    last_applied_sequence   BIGINT,
    last_applied_partition  VARCHAR(128),
    correlation_id          UUID,
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_landing_content_projection PRIMARY KEY (content_kind, schema_version_major),
    CONSTRAINT valid_offset CHECK (last_applied_offset >= 0),
    CONSTRAINT valid_sequence CHECK (
        last_applied_sequence IS NULL OR last_applied_sequence >= 0
    )
);

CREATE INDEX IF NOT EXISTS idx_landing_content_projection_kind
    ON landing_content_projection (content_kind);

CREATE INDEX IF NOT EXISTS idx_landing_content_projection_updated_at
    ON landing_content_projection (updated_at);

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '068',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
