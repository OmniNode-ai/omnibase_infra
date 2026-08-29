-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback for 101 (OMN-16923): re-narrow savings_estimates to the flat 074
-- shape in the SERVICE database.
--
-- ============================================================================
-- THIS DIRECTION IS NOT SYMMETRIC WITH THE FORWARD
-- ============================================================================
-- Forward 101 is a pure widening, so it can never fail on data. This file is
-- the NARROWING, and a narrowing is only safe against the rows that happen to
-- be present when it runs. Every affected column therefore proves the fit in
-- its own USING clause, per row, and aborts the statement with a message naming
-- this ticket rather than letting a value be truncated.
--
-- Same two constraints as the forward file, for the same reasons:
--   * no procedural block (`check_application_database_sql.py` rejects every
--     `DO` in changed SQL), so the refusal is a CAST of the explanation;
--   * each message is anchored to the column (`|| left(<col>, 0)`, appending
--     nothing) so it is not a CONSTANT -- PostgreSQL resolves constant casts at
--     PARSE time, and a bare `'...'::NUMERIC` in an untaken branch would abort
--     on every database rather than only the offending one.
--
-- The cast target is NUMERIC even in the varchar branches, with a `::TEXT` back
-- to the branch type. `'...'::VARCHAR(n)` would NOT do: an explicit cast to a
-- length-limited character type TRUNCATES silently instead of erroring, so the
-- "guard" would have written the message into the row and reported success.
--
-- The three CHECK constraints 101 added are dropped by their declared names
-- only. The flat file's own `non_negative_local` / `non_negative_cloud` /
-- `savings_consistency` are NOT touched -- they are 074's, not 101's, and the
-- identical predicates stay enforced after this file runs.
--
-- The whole file runs in one explicit transaction, for the same reason the
-- forward file does: the runner does not pass `--single-transaction`, so a
-- later refusal must not leave an earlier narrowing committed.
--
-- Ticket: OMN-16923.

BEGIN;

ALTER TABLE savings_estimates
    DROP CONSTRAINT IF EXISTS savings_estimates_amounts_match;
ALTER TABLE savings_estimates
    DROP CONSTRAINT IF EXISTS savings_estimates_cloud_cost_usd_check;
ALTER TABLE savings_estimates
    DROP CONSTRAINT IF EXISTS savings_estimates_local_cost_usd_check;

ALTER TABLE savings_estimates
    ALTER COLUMN session_id TYPE VARCHAR(255)
        USING CASE
            WHEN length(session_id) > 255 THEN
                ('OMN-16923 rollback: refusing to narrow savings_estimates.session_id '
                 'to VARCHAR(255) -- a stored value is longer and would be '
                 'TRUNCATED. Nothing was applied.'
                 || left(session_id, 0))::NUMERIC::TEXT
            ELSE session_id
        END,
    ALTER COLUMN model_local TYPE VARCHAR(255)
        USING CASE
            WHEN length(model_local) > 255 THEN
                ('OMN-16923 rollback: refusing to narrow savings_estimates.model_local '
                 'to VARCHAR(255) -- a stored value is longer and would be '
                 'TRUNCATED. Nothing was applied.'
                 || left(model_local, 0))::NUMERIC::TEXT
            ELSE model_local
        END,
    ALTER COLUMN model_cloud_baseline TYPE VARCHAR(255)
        USING CASE
            WHEN length(model_cloud_baseline) > 255 THEN
                ('OMN-16923 rollback: refusing to narrow savings_estimates.model_cloud_baseline '
                 'to VARCHAR(255) -- a stored value is longer and would be '
                 'TRUNCATED. Nothing was applied.'
                 || left(model_cloud_baseline, 0))::NUMERIC::TEXT
            ELSE model_cloud_baseline
        END,
    ALTER COLUMN repo_name TYPE VARCHAR(255)
        USING CASE
            WHEN length(repo_name) > 255 THEN
                ('OMN-16923 rollback: refusing to narrow savings_estimates.repo_name '
                 'to VARCHAR(255) -- a stored value is longer and would be '
                 'TRUNCATED. Nothing was applied.'
                 || left(repo_name, 0))::NUMERIC::TEXT
            ELSE repo_name
        END,
    ALTER COLUMN machine_id TYPE VARCHAR(64)
        USING CASE
            WHEN length(machine_id) > 64 THEN
                ('OMN-16923 rollback: refusing to narrow savings_estimates.machine_id '
                 'to VARCHAR(64) -- a stored value is longer and would be '
                 'TRUNCATED. Nothing was applied.'
                 || left(machine_id, 0))::NUMERIC::TEXT
            ELSE machine_id
        END;

-- NUMERIC(14,6) admits 8 integer digits. The scale is unchanged, so rounding is
-- impossible here; only magnitude can fail.
ALTER TABLE savings_estimates
    ALTER COLUMN local_cost_usd TYPE NUMERIC(14, 6)
        USING CASE
            WHEN abs(local_cost_usd) >= 100000000 THEN
                ('OMN-16923 rollback: refusing to narrow savings_estimates.local_cost_usd '
                 'to NUMERIC(14,6) -- a stored value needs more than the 8 integer '
                 'digits that target admits. Nothing was applied.'
                 || left(local_cost_usd::TEXT, 0))::NUMERIC
            ELSE local_cost_usd
        END,
    ALTER COLUMN cloud_cost_usd TYPE NUMERIC(14, 6)
        USING CASE
            WHEN abs(cloud_cost_usd) >= 100000000 THEN
                ('OMN-16923 rollback: refusing to narrow savings_estimates.cloud_cost_usd '
                 'to NUMERIC(14,6) -- a stored value needs more than the 8 integer '
                 'digits that target admits. Nothing was applied.'
                 || left(cloud_cost_usd::TEXT, 0))::NUMERIC
            ELSE cloud_cost_usd
        END,
    ALTER COLUMN savings_usd TYPE NUMERIC(14, 6)
        USING CASE
            WHEN abs(savings_usd) >= 100000000 THEN
                ('OMN-16923 rollback: refusing to narrow savings_estimates.savings_usd '
                 'to NUMERIC(14,6) -- a stored value needs more than the 8 integer '
                 'digits that target admits. Nothing was applied.'
                 || left(savings_usd::TEXT, 0))::NUMERIC
            ELSE savings_usd
        END;

COMMIT;
