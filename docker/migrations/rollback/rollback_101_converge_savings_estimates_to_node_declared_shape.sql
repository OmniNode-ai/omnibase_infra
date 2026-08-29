-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback for 101 (OMN-16923): re-narrow public.savings_estimates to the flat
-- 074 shape in the SERVICE database.
--
-- ============================================================================
-- THIS DIRECTION IS NOT SYMMETRIC WITH THE FORWARD
-- ============================================================================
-- Forward 101 is a pure widening, so it can never fail on data. This file is
-- the NARROWING, and a narrowing is only safe against the rows that happen to
-- be present when it runs. It therefore proves the fit for every affected
-- column before it alters anything, and RAISEs with the offending count rather
-- than letting Postgres truncate a value or reject a row halfway through.
--
-- The three CHECK constraints 101 added are dropped by their declared names
-- only. The flat file's own `non_negative_local` / `non_negative_cloud` /
-- `savings_consistency` are NOT touched -- they are 074's, not 101's, and the
-- identical predicates stay enforced after this file runs.
--
-- Ticket: OMN-16923.

DO $omn16923_rollback$
DECLARE
    v_rel        regclass := to_regclass('public.savings_estimates');
    v_col        TEXT;
    v_limit      INT;
    v_violations BIGINT;
    v_varchar_columns CONSTANT TEXT[] := ARRAY[
        'session_id', 'model_local', 'model_cloud_baseline', 'repo_name', 'machine_id'
    ];
    v_varchar_limits CONSTANT INT[] := ARRAY[255, 255, 255, 255, 64];
    v_numeric_columns CONSTANT TEXT[] := ARRAY[
        'local_cost_usd', 'cloud_cost_usd', 'savings_usd'
    ];
BEGIN
    IF v_rel IS NULL THEN
        RAISE NOTICE
            'OMN-16923 rollback: public.savings_estimates does not exist here -- nothing to do.';
        RETURN;
    END IF;

    ALTER TABLE public.savings_estimates
        DROP CONSTRAINT IF EXISTS savings_estimates_amounts_match;
    ALTER TABLE public.savings_estimates
        DROP CONSTRAINT IF EXISTS savings_estimates_cloud_cost_usd_check;
    ALTER TABLE public.savings_estimates
        DROP CONSTRAINT IF EXISTS savings_estimates_local_cost_usd_check;

    FOR i IN 1 .. array_length(v_varchar_columns, 1)
    LOOP
        v_col := v_varchar_columns[i];
        v_limit := v_varchar_limits[i];
        EXECUTE format(
            'SELECT count(*) FROM public.savings_estimates WHERE length(%I) > %s',
            v_col, v_limit
        ) INTO v_violations;
        IF v_violations > 0 THEN
            RAISE EXCEPTION
                'OMN-16923 rollback: refusing to narrow public.savings_estimates.% to '
                'VARCHAR(%) -- % row(s) already exceed that length and would be '
                'truncated. Nothing was applied.', v_col, v_limit, v_violations;
        END IF;
        EXECUTE format(
            'ALTER TABLE public.savings_estimates ALTER COLUMN %I TYPE VARCHAR(%s)',
            v_col, v_limit
        );
    END LOOP;

    FOREACH v_col IN ARRAY v_numeric_columns
    LOOP
        -- NUMERIC(14,6) admits 8 integer digits. abs(value) >= 10^8 would be
        -- rejected by the cast; the scale is unchanged so no rounding is possible.
        EXECUTE format(
            'SELECT count(*) FROM public.savings_estimates WHERE abs(%I) >= 100000000',
            v_col
        ) INTO v_violations;
        IF v_violations > 0 THEN
            RAISE EXCEPTION
                'OMN-16923 rollback: refusing to narrow public.savings_estimates.% to '
                'NUMERIC(14,6) -- % row(s) hold a value needing more than 8 integer '
                'digits. Nothing was applied.', v_col, v_violations;
        END IF;
        EXECUTE format(
            'ALTER TABLE public.savings_estimates ALTER COLUMN %I TYPE NUMERIC(14, 6)',
            v_col
        );
    END LOOP;
END
$omn16923_rollback$;
