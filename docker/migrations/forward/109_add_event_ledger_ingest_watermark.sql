-- OMN-19803: commit-order replay bounds for captured event_ledger rows.
-- Historical rows remain NULL: their write order cannot be reconstructed.
ALTER TABLE public.event_ledger
    ADD COLUMN IF NOT EXISTS ingest_watermark BIGINT;

DO $$
DECLARE
    v_definition TEXT;
    v_validated BOOLEAN;
BEGIN
    SELECT
        regexp_replace(
            lower(pg_get_constraintdef(constraint_row.oid)),
            '[[:space:]()]',
            '',
            'g'
        ),
        constraint_row.convalidated
    INTO v_definition, v_validated
    FROM pg_constraint AS constraint_row
    WHERE constraint_row.conrelid = 'public.event_ledger'::regclass
      AND constraint_row.conname = 'event_ledger_ingest_watermark_positive';

    IF NOT FOUND THEN
        ALTER TABLE public.event_ledger
            ADD CONSTRAINT event_ledger_ingest_watermark_positive
            CHECK (ingest_watermark IS NULL OR ingest_watermark > 0);
    ELSIF v_definition <> 'checkingest_watermarkisnulloringest_watermark>0'
       OR v_validated IS NOT TRUE THEN
        RAISE EXCEPTION
            'event_ledger_ingest_watermark_positive exists with a noncanonical definition';
    END IF;
END;
$$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_event_ledger_ingest_watermark
    ON public.event_ledger (topic, partition, ingest_watermark)
    WHERE ingest_watermark IS NOT NULL;

CREATE TABLE IF NOT EXISTS public.event_ledger_ingest_watermark_counter (
    topic TEXT NOT NULL,
    partition INTEGER NOT NULL,
    next_watermark BIGINT NOT NULL CHECK (next_watermark > 0),
    PRIMARY KEY (topic, partition)
);

-- The counter row serializes writers on one Kafka partition. It is advanced
-- only when the idempotency-key INSERT succeeds, in the same transaction.
CREATE OR REPLACE FUNCTION public.append_event_ledger_with_watermark(
    p_topic TEXT,
    p_partition INTEGER,
    p_kafka_offset BIGINT,
    p_event_key BYTEA,
    p_event_value BYTEA,
    p_onex_headers JSONB,
    p_envelope_id UUID,
    p_correlation_id UUID,
    p_event_type TEXT,
    p_source TEXT,
    p_event_timestamp TIMESTAMPTZ
)
RETURNS TABLE (ledger_entry_id UUID, ingest_watermark BIGINT, duplicate BOOLEAN)
LANGUAGE plpgsql
AS $$
DECLARE
    v_next_watermark BIGINT;
    v_ledger_entry_id UUID;
    v_ingest_watermark BIGINT;
BEGIN
    INSERT INTO public.event_ledger_ingest_watermark_counter
        (topic, partition, next_watermark)
    VALUES (p_topic, p_partition, 1)
    ON CONFLICT (topic, partition) DO NOTHING;

    SELECT counter.next_watermark INTO v_next_watermark
    FROM public.event_ledger_ingest_watermark_counter AS counter
    WHERE counter.topic = p_topic AND counter.partition = p_partition
    FOR UPDATE;

    INSERT INTO public.event_ledger (
        topic, partition, kafka_offset, event_key, event_value, onex_headers,
        envelope_id, correlation_id, event_type, source, event_timestamp,
        ingest_watermark
    ) VALUES (
        p_topic, p_partition, p_kafka_offset, p_event_key, p_event_value,
        p_onex_headers, p_envelope_id, p_correlation_id, p_event_type,
        p_source, p_event_timestamp, v_next_watermark
    )
    ON CONFLICT (topic, partition, kafka_offset) DO NOTHING
    RETURNING event_ledger.ledger_entry_id, event_ledger.ingest_watermark
    INTO v_ledger_entry_id, v_ingest_watermark;

    IF v_ledger_entry_id IS NOT NULL THEN
        UPDATE public.event_ledger_ingest_watermark_counter AS counter
        SET next_watermark = v_next_watermark + 1
        WHERE counter.topic = p_topic AND counter.partition = p_partition;
        RETURN QUERY SELECT v_ledger_entry_id, v_ingest_watermark, FALSE;
        RETURN;
    END IF;

    SELECT event_ledger.ledger_entry_id, event_ledger.ingest_watermark
    INTO v_ledger_entry_id, v_ingest_watermark
    FROM public.event_ledger
    WHERE event_ledger.topic = p_topic
      AND event_ledger.partition = p_partition
      AND event_ledger.kafka_offset = p_kafka_offset;
    RETURN QUERY SELECT v_ledger_entry_id, v_ingest_watermark, TRUE;
END;
$$;
