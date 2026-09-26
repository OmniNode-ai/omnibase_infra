-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-19728: sim-202 archive replay outbox in the omnibase_infra database.
-- Ordinal 107 is retired/burned (OMN-17486); 108 is the next safe number.
-- The flat migration runner and OMNIBASE_INFRA_DB_URL pool target this DB.

CREATE TABLE IF NOT EXISTS public.sim_archive_rehydration_outbox (
    source_topic TEXT NOT NULL,
    source_partition INTEGER NOT NULL CHECK (source_partition >= 0),
    source_offset BIGINT NOT NULL CHECK (source_offset >= 0),
    target_topic TEXT NOT NULL,
    record_key BYTEA,
    record_value BYTEA NOT NULL,
    headers_json TEXT NOT NULL,
    timestamp_ms BIGINT NOT NULL CHECK (timestamp_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,
    CONSTRAINT pk_sim_archive_rehydration_outbox
        PRIMARY KEY (source_topic, source_partition, source_offset),
    CONSTRAINT ck_sim_archive_rehydration_target_matches_source
        CHECK (target_topic = source_topic)
);

CREATE INDEX IF NOT EXISTS ix_sim_archive_rehydration_outbox_pending
    ON public.sim_archive_rehydration_outbox
       (source_topic, source_partition, source_offset)
    WHERE delivered_at IS NULL;

COMMENT ON TABLE public.sim_archive_rehydration_outbox IS
    'OMN-19728 sim-only source-coordinate deduplication and at-least-once replay relay; never an archive inventory or production restore ledger.';
