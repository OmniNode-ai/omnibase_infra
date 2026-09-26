-- OMN-19728: writer-assigned, commit-ordered ingest cursor for event_ledger.
-- Historical rows remain unversioned. Kafka offsets and ledger_written_at do
-- not establish historical ledger commit order.

ALTER TABLE public.event_ledger
    ADD COLUMN IF NOT EXISTS ingest_epoch SMALLINT,
    ADD COLUMN IF NOT EXISTS ingest_seq BIGINT;

ALTER TABLE public.event_ledger
    ADD CONSTRAINT event_ledger_ingest_cursor_pair_chk
    CHECK (
        (ingest_epoch IS NULL AND ingest_seq IS NULL)
        OR (ingest_epoch IS NOT NULL AND ingest_seq IS NOT NULL
            AND ingest_epoch = 1 AND ingest_seq > 0)
    );

CREATE UNIQUE INDEX IF NOT EXISTS uq_event_ledger_ingest_cursor
    ON public.event_ledger (topic, partition, ingest_epoch, ingest_seq)
    WHERE ingest_seq IS NOT NULL;

CREATE TABLE IF NOT EXISTS public.ledger_partition_clock (
    topic TEXT NOT NULL,
    partition INTEGER NOT NULL,
    last_ingest_seq BIGINT NOT NULL DEFAULT 0 CHECK (last_ingest_seq >= 0),
    PRIMARY KEY (topic, partition)
);

-- Diagnostic cutover snapshot only. It does not prove historical prefix
-- completeness, nor prevent an old writer from appending NULL/NULL later.
CREATE TABLE IF NOT EXISTS public.ledger_ingest_legacy_baseline (
    topic TEXT NOT NULL,
    partition INTEGER NOT NULL,
    legacy_max_kafka_offset BIGINT NOT NULL,
    legacy_row_count BIGINT NOT NULL CHECK (legacy_row_count >= 0),
    captured_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (topic, partition)
);

INSERT INTO public.ledger_ingest_legacy_baseline (
    topic, partition, legacy_max_kafka_offset, legacy_row_count
)
SELECT topic, partition, MAX(kafka_offset), COUNT(*)
FROM public.event_ledger
WHERE ingest_epoch IS NULL AND ingest_seq IS NULL
GROUP BY topic, partition
ON CONFLICT (topic, partition) DO NOTHING;

COMMENT ON COLUMN public.event_ledger.ingest_seq IS
    'Commit-ordered per-(topic,partition) sequence assigned by append_event_ledger_with_watermark; NULL means unversioned legacy/old-writer row.';
COMMENT ON TABLE public.ledger_ingest_legacy_baseline IS
    'Migration-time diagnostic snapshot only; never a historical completeness or ordering claim.';

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
RETURNS TABLE (
    ledger_entry_id UUID,
    duplicate BOOLEAN,
    ingest_epoch SMALLINT,
    ingest_seq BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_existing public.event_ledger%ROWTYPE;
    v_next_seq BIGINT;
    v_entry_id UUID;
BEGIN
    IF p_partition < 0 OR p_kafka_offset < 0 THEN
        RAISE EXCEPTION 'ledger partition and offset must be nonnegative'
            USING ERRCODE = '22023';
    END IF;

    -- The upsert locks one partition clock row until the caller's transaction
    -- commits. A second writer cannot allocate/commit a later sequence first.
    INSERT INTO public.ledger_partition_clock (topic, partition, last_ingest_seq)
    VALUES (p_topic, p_partition, 0)
    ON CONFLICT (topic, partition) DO UPDATE
        SET last_ingest_seq = public.ledger_partition_clock.last_ingest_seq;

    SELECT el.* INTO v_existing
    FROM public.event_ledger AS el
    WHERE el.topic = p_topic
      AND el.partition = p_partition
      AND el.kafka_offset = p_kafka_offset;

    IF FOUND THEN
        IF v_existing.event_key IS DISTINCT FROM p_event_key
           OR v_existing.event_value IS DISTINCT FROM p_event_value
           OR v_existing.onex_headers IS DISTINCT FROM p_onex_headers
           OR v_existing.envelope_id IS DISTINCT FROM p_envelope_id
           OR v_existing.correlation_id IS DISTINCT FROM p_correlation_id
           OR v_existing.event_type IS DISTINCT FROM p_event_type
           OR v_existing.source IS DISTINCT FROM p_source
           OR v_existing.event_timestamp IS DISTINCT FROM p_event_timestamp THEN
            RAISE EXCEPTION 'conflicting event at existing ledger position'
                USING ERRCODE = '23505';
        END IF;
        RETURN QUERY SELECT v_existing.ledger_entry_id, TRUE,
                            v_existing.ingest_epoch, v_existing.ingest_seq;
        RETURN;
    END IF;

    UPDATE public.ledger_partition_clock AS clock
    SET last_ingest_seq = clock.last_ingest_seq + 1
    WHERE clock.topic = p_topic AND clock.partition = p_partition
    RETURNING clock.last_ingest_seq INTO v_next_seq;

    INSERT INTO public.event_ledger (
        topic, partition, kafka_offset, event_key, event_value, onex_headers,
        envelope_id, correlation_id, event_type, source, event_timestamp,
        ingest_epoch, ingest_seq
    ) VALUES (
        p_topic, p_partition, p_kafka_offset, p_event_key, p_event_value,
        p_onex_headers, p_envelope_id, p_correlation_id, p_event_type,
        p_source, p_event_timestamp, 1, v_next_seq
    ) RETURNING public.event_ledger.ledger_entry_id INTO v_entry_id;

    RETURN QUERY SELECT v_entry_id, FALSE, 1::SMALLINT, v_next_seq;
END;
$$;
