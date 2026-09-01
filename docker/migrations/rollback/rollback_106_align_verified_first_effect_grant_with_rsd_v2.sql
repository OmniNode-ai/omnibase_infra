-- Refuse a lossy downgrade if a public RSD v2 grant has been persisted.

DO $block$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM public.first_effect_verified_grant_ledger
         WHERE retry_disposition = 'forbidden'
    ) THEN
        RAISE EXCEPTION
            'cannot roll back migration 106 while RSD v2 verified grants exist';
    END IF;
END;
$block$;

ALTER TABLE public.first_effect_verified_grant_ledger
    DROP CONSTRAINT ck_verified_first_effect_retry_disposition,
    ADD CONSTRAINT ck_verified_first_effect_retry_disposition
        CHECK (retry_disposition = 'never-republish-after-ambiguous.v1'),
    DROP CONSTRAINT ck_verified_first_effect_expected_topic,
    ADD CONSTRAINT ck_verified_first_effect_expected_topic
        CHECK (expected_output_topic ~ '^onex\.[a-z0-9-]+(\.[a-z0-9-]+)+\.v[1-9][0-9]*$');
