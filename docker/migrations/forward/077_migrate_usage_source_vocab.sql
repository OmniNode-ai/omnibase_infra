-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT

-- OMN-10382: migrate LLM usage_source values to EnumUsageSource vocabulary.

BEGIN;

CREATE TYPE usage_source_type_v2 AS ENUM ('measured', 'estimated', 'unknown');

ALTER TABLE llm_call_metrics
    ALTER COLUMN usage_source DROP DEFAULT;

ALTER TABLE llm_call_metrics
    ALTER COLUMN usage_source TYPE usage_source_type_v2
    USING CASE usage_source::text
        WHEN 'API' THEN 'measured'
        WHEN 'ESTIMATED' THEN 'estimated'
        WHEN 'MISSING' THEN 'unknown'
        WHEN 'measured' THEN 'measured'
        WHEN 'estimated' THEN 'estimated'
        WHEN 'unknown' THEN 'unknown'
        ELSE 'unknown'
    END::usage_source_type_v2;

ALTER TABLE llm_call_metrics
    ALTER COLUMN usage_source SET DEFAULT 'unknown';

ALTER TABLE llm_call_metrics
    ALTER COLUMN compute_usage_source TYPE usage_source_type_v2
    USING CASE compute_usage_source::text
        WHEN 'API' THEN 'measured'
        WHEN 'ESTIMATED' THEN 'estimated'
        WHEN 'MISSING' THEN 'unknown'
        WHEN 'measured' THEN 'measured'
        WHEN 'estimated' THEN 'estimated'
        WHEN 'unknown' THEN 'unknown'
        ELSE NULL
    END::usage_source_type_v2;

DROP TYPE usage_source_type;

ALTER TYPE usage_source_type_v2 RENAME TO usage_source_type;

COMMENT ON TYPE usage_source_type IS
    'Shared usage-source vocabulary: measured, estimated, unknown.';

COMMENT ON COLUMN llm_call_metrics.usage_source IS
    'Usage provenance using the shared EnumUsageSource vocabulary.';

COMMENT ON COLUMN llm_call_metrics.compute_usage_source IS
    'GPU usage provenance using the shared EnumUsageSource vocabulary.';

COMMIT;
