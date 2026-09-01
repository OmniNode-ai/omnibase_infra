-- Align the immutable verified-grant projection with public RSD v2 wire pins.
--
-- Migration 105 remains immutable. Its original ONEX-topic and retry checks
-- are widened only for new RSD v2 grants; existing ONEX records remain valid.
-- The topic expression validates shape only. Admission remains the fixed
-- signature check plus composition-owned deployment output-pin equality.
-- No payload, signature, key, broker, or publisher data is introduced.

ALTER TABLE public.first_effect_verified_grant_ledger
    DROP CONSTRAINT ck_verified_first_effect_retry_disposition,
    ADD CONSTRAINT ck_verified_first_effect_retry_disposition
        CHECK (
            retry_disposition IN (
                'never-republish-after-ambiguous.v1',
                'forbidden'
            )
        ),
    DROP CONSTRAINT ck_verified_first_effect_expected_topic,
    ADD CONSTRAINT ck_verified_first_effect_expected_topic
        CHECK (expected_output_topic ~ '^[a-z][a-z0-9-]*(\.[a-z][a-z0-9-]*)+$');
