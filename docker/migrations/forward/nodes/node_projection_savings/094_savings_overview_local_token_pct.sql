-- OMN-20320: projection_cost_savings_overview.local_token_pct was a literal
-- `0::float` (077 shipped it, 089 kept it and added a warning saying so), so the
-- dashboard read 0% local while projection_delegation_savings_series, over the
-- same rows, read 0.88 to 0.98 local share per day.
--
-- The premise recorded in 089 -- "nothing measures a local/cloud token split" --
-- is false: `delegation_events.cost_tier_name` records the tier that served each
-- run, and the series view already buckets it (local / cheap / premium).
-- This replacement measures the split with the SAME bucketing as the series:
--
--   local   : cost_tier_name = 'local'
--   counted : cost_tier_name IN ('local', 'cheap_cloud', 'cheap_frontier',
--                                'claude')
--
--   local_token_pct = SUM(prompt + completion tokens of local runs)
--                     / SUM(prompt + completion tokens of counted runs)
--
-- A run with no recorded tier (savings_estimates-only rows carry none) is
-- outside the denominator, exactly as it is outside the series' tier shares,
-- rather than being counted as cloud. When no run has a tier the value stays 0
-- and `warnings` says the split is unmeasured, which keeps a measured zero
-- distinguishable from an absent measurement.
--
-- The series' share is per run COUNT; this one is per TOKEN, because the column
-- is named for tokens. They agree whenever local and non-local runs are the same
-- size and differ by run size otherwise.
--
-- Output columns are unchanged in name, type and order, so CREATE OR REPLACE is
-- legal, grants survive, and security_invoker is set again at the end (a REPLACE
-- resets omitted options -- OMN-19808).

CREATE OR REPLACE VIEW public.projection_cost_savings_overview AS
WITH raw_savings_runs AS (
    SELECT
        -- OMN-17426: the house tenant is spelled differently by the two
        -- sources and this is where they are reconciled.
        -- `savings_estimates.tenant_id` is TEXT and stores the house SLUG
        -- 'omninode'; `delegation_events.tenant_id` is uuid (that node's
        -- migration 0034) and stores the house UUID. Left alone, one
        -- logical tenant becomes TWO groups in this view, the writer can
        -- only ever republish one of them, and the other silently never
        -- reaches the page. Worse, the writer's re-read binds the slug as
        -- `app.tenant_id` and `delegation_events`' policy casts that GUC to
        -- uuid, so the read does not return the wrong rows -- it ABORTS
        -- with `invalid input syntax for type uuid: "omninode"`, which is
        -- the same failure node_projection_delegation hit under OMN-18139.
        --
        -- The UUID is not invented here: it is the platform's own
        -- `HOUSE_TENANT_UUID`, the value `_house_tenant_interim_default`
        -- already stamps for uuid-converted tables, and the same rekey
        -- node_projection_delegation/0030 performed for delegation_budget_state.
        -- Every other tenant value is already a UUID in both columns and
        -- passes through untouched.
        CASE WHEN tenant_id = 'omninode' THEN '820272f9-4aaf-5add-a2df-0af942852ab2'
             ELSE tenant_id::text END AS tenant_id,
        -- onex-api joins `savings_estimates.session_id` to
        -- `delegation_events.correlation_id`; this view composes the same two
        -- sources on the same key, so the id a reader matches on is the same
        -- id the API matched on.
        session_id AS correlation_id,
        session_id,
        COALESCE(NULLIF(task_type, ''), 'savings-estimated') AS task_type,
        COALESCE(NULLIF(model_local, ''), 'local') AS model_id,
        COALESCE(NULLIF(model_local, ''), 'Local model') AS display_name,
        local_cost_usd::float AS cost_usd,
        cloud_cost_usd::float AS baseline_cost_usd,
        savings_usd::float AS savings_usd,
        COALESCE(prompt_tokens, 0)::int AS prompt_tokens,
        COALESCE(completion_tokens, 0)::int AS completion_tokens,
        NULL::int AS tokens_to_compliance,
        NULL::int AS latency_ms,
        NULL::boolean AS quality_gate_passed,
        NULL::text AS cost_tier_name,
        COALESCE(usage_source, 'unknown') AS token_provenance,
        COALESCE(updated_at, created_at, event_timestamp)::timestamptz
            AS projected_at
    FROM public.savings_estimates
    WHERE tenant_id IS NOT NULL
      AND session_id IS NOT NULL
),
savings_runs AS (
    SELECT
        tenant_id, correlation_id, session_id, task_type, model_id, display_name,
        cost_usd, baseline_cost_usd, savings_usd, prompt_tokens,
        completion_tokens, tokens_to_compliance, latency_ms, quality_gate_passed,
        cost_tier_name, token_provenance, projected_at
    FROM (
        SELECT
            raw_savings_runs.*,
            ROW_NUMBER() OVER (
                PARTITION BY tenant_id, correlation_id
                ORDER BY projected_at DESC, session_id DESC
            ) AS tenant_run_rank
        FROM raw_savings_runs
    ) ranked
    WHERE tenant_run_rank = 1
),
event_runs AS (
    SELECT
        tenant_id::text AS tenant_id,
        COALESCE(NULLIF(correlation_id, ''), NULLIF(session_id, ''), id::text)
            AS correlation_id,
        COALESCE(NULLIF(session_id, ''), NULLIF(correlation_id, ''), id::text)
            AS session_id,
        COALESCE(NULLIF(task_type, ''), 'delegation') AS task_type,
        COALESCE(NULLIF(model_name, ''), NULLIF(delegated_to, ''), 'local')
            AS model_id,
        COALESCE(NULLIF(model_name, ''), NULLIF(delegated_to, ''), 'Local model')
            AS display_name,
        COALESCE(cost_usd, 0)::float AS cost_usd,
        (COALESCE(cost_usd, 0) + COALESCE(cost_savings_usd, 0))::float
            AS baseline_cost_usd,
        COALESCE(cost_savings_usd, 0)::float AS savings_usd,
        COALESCE(tokens_input, 0)::int AS prompt_tokens,
        COALESCE(tokens_output, 0)::int AS completion_tokens,
        NULLIF(tokens_to_compliance, 0)::int AS tokens_to_compliance,
        COALESCE(delegation_latency_ms, latency_ms)::int AS latency_ms,
        quality_gate_passed,
        cost_tier_name::text AS cost_tier_name,
        CASE
            WHEN cost_measurement_source IN (
                'metered', 'free_local', 'budgeted_in_budget',
                'budgeted_overage', 'budgeted_split') THEN 'measured'
            WHEN cost_measurement_source = 'manifest_compute' THEN 'estimated'
            ELSE 'unknown'
        END AS token_provenance,
        COALESCE(created_at, timestamp)::timestamptz AS projected_at
    FROM public.delegation_events
    WHERE tenant_id IS NOT NULL
),
combined_runs AS (
    -- Delegation rows own run identity. The savings row wins only for the
    -- savings-derived numeric fields, matching onex-api's field-level
    -- COALESCE shape instead of dropping the whole delegation row.
    SELECT
        event_runs.tenant_id,
        event_runs.correlation_id,
        event_runs.session_id,
        event_runs.task_type,
        event_runs.model_id,
        event_runs.display_name,
        COALESCE(savings_runs.cost_usd, event_runs.cost_usd) AS cost_usd,
        COALESCE(savings_runs.baseline_cost_usd, event_runs.baseline_cost_usd)
            AS baseline_cost_usd,
        COALESCE(savings_runs.savings_usd, event_runs.savings_usd)
            AS savings_usd,
        COALESCE(NULLIF(savings_runs.prompt_tokens, 0), event_runs.prompt_tokens)
            AS prompt_tokens,
        COALESCE(
            NULLIF(savings_runs.completion_tokens, 0),
            event_runs.completion_tokens
        ) AS completion_tokens,
        event_runs.tokens_to_compliance,
        event_runs.latency_ms,
        event_runs.quality_gate_passed,
        event_runs.cost_tier_name,
        COALESCE(NULLIF(event_runs.token_provenance, 'unknown'),
                 savings_runs.token_provenance,
                 'unknown') AS token_provenance,
        COALESCE(
            GREATEST(event_runs.projected_at, savings_runs.projected_at),
            event_runs.projected_at,
            savings_runs.projected_at
        )
            AS projected_at
    FROM event_runs
    LEFT JOIN savings_runs
      ON savings_runs.correlation_id = event_runs.correlation_id
     AND savings_runs.tenant_id = event_runs.tenant_id
    UNION ALL
    SELECT savings_runs.*
    FROM savings_runs
    WHERE NOT EXISTS (
        SELECT 1
        FROM event_runs
        WHERE event_runs.correlation_id = savings_runs.correlation_id
          AND event_runs.tenant_id = savings_runs.tenant_id
    )
),
totals AS (
    SELECT
        tenant_id,
        COALESCE(SUM(cost_usd), 0)::float AS total_cost_usd,
        COALESCE(SUM(baseline_cost_usd), 0)::float AS total_baseline_cost_usd,
        COALESCE(SUM(savings_usd), 0)::float AS total_savings_usd,
        COALESCE(SUM(prompt_tokens + completion_tokens), 0)::int AS tokens_total,
        COALESCE(SUM(COALESCE(tokens_to_compliance, 0)), 0)::int
            AS tokens_to_compliance,
        COUNT(*) FILTER (
            WHERE prompt_tokens + completion_tokens > 0
        )::int AS measured_run_count,
        COUNT(*) FILTER (
            WHERE prompt_tokens + completion_tokens = 0
        )::int AS zero_token_run_count,
        COUNT(*)::int AS run_count,
        COALESCE(SUM(prompt_tokens + completion_tokens) FILTER (
            WHERE cost_tier_name = 'local'
        ), 0)::float AS local_tier_tokens,
        COALESCE(SUM(prompt_tokens + completion_tokens) FILTER (
            WHERE cost_tier_name IN (
                'local', 'cheap_cloud', 'cheap_frontier', 'claude'
            )
        ), 0)::float AS tiered_tokens,
        MAX(projected_at) AS latest_projection_updated_at
    FROM combined_runs
    GROUP BY tenant_id
),
model_rows AS (
    SELECT
        tenant_id,
        COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'model_id', model_id,
                    'display_name', display_name,
                    'execution_mode', 'delegated',
                    'task_count', task_count,
                    'tokens_total', tokens_total,
                    'cost_usd', cost_usd,
                    'baseline_cost_usd', baseline_cost_usd,
                    'savings_usd', savings_usd,
                    'savings_pct', CASE WHEN baseline_cost_usd > 0
                        THEN savings_usd / baseline_cost_usd ELSE 0 END,
                    'runtime_address', NULL,
                    'evidence_ref', NULL
                )
                ORDER BY savings_usd DESC, display_name
            ),
            '[]'::jsonb
        ) AS rows
    FROM (
        SELECT
            tenant_id,
            model_id,
            display_name,
            COUNT(*)::int AS task_count,
            COALESCE(SUM(prompt_tokens + completion_tokens), 0)::int
                AS tokens_total,
            COALESCE(SUM(cost_usd), 0)::float AS cost_usd,
            COALESCE(SUM(baseline_cost_usd), 0)::float AS baseline_cost_usd,
            COALESCE(SUM(savings_usd), 0)::float AS savings_usd
        FROM combined_runs
        GROUP BY tenant_id, model_id, display_name
    ) grouped_models
    GROUP BY tenant_id
),
ranked_runs AS (
    SELECT
        combined_runs.*,
        ROW_NUMBER() OVER (
            PARTITION BY tenant_id
            ORDER BY projected_at DESC, correlation_id DESC, session_id DESC
        ) AS tenant_rank
    FROM combined_runs
),
recent_runs AS (
    SELECT
        tenant_id,
        COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    -- Pre-existing keys, unchanged in name and meaning.
                    'session_id', session_id,
                    'task_type', task_type,
                    'model_name', display_name,
                    'prompt_tokens', prompt_tokens,
                    'completion_tokens', completion_tokens,
                    'total_tokens', prompt_tokens + completion_tokens,
                    'savings_usd', savings_usd,
                    'latency_ms', latency_ms,
                    'created_at', projected_at,
                    'token_provenance', token_provenance,
                    -- OMN-17426, added beside them: what identifies the run,
                    -- what it cost, and whether it passed its gate.
                    'correlation_id', correlation_id,
                    'cost_usd', cost_usd,
                    'cost_savings_usd', savings_usd,
                    'quality_gate_passed', quality_gate_passed,
                    'tokens_to_compliance', tokens_to_compliance,
                    'cost_tier_name', cost_tier_name
                )
                ORDER BY projected_at DESC, correlation_id DESC, session_id DESC
            ),
            '[]'::jsonb
        ) AS rows
    FROM ranked_runs
    WHERE tenant_rank <= 20
    GROUP BY tenant_id
),
warnings AS (
    SELECT
        tenant_id,
        (
            CASE WHEN zero_token_run_count > 0 THEN jsonb_build_array(
                zero_token_run_count
                || ' run(s) carry no measured served-token counts; they are'
                || ' counted in cost, savings, and compliance-token totals'
            ) ELSE '[]'::jsonb END
            ||
            CASE WHEN tokens_total > 0 AND tiered_tokens = 0 THEN jsonb_build_array(
                'No run carries a cost tier, so the local/cloud token split is'
                || ' unmeasured and local_token_pct is reported as zero'
            ) ELSE '[]'::jsonb END
        ) AS rows
    FROM totals
)
SELECT
    'all'::text AS "window",
    totals.total_cost_usd,
    totals.total_baseline_cost_usd,
    totals.total_savings_usd,
    CASE WHEN totals.total_baseline_cost_usd > 0
        THEN totals.total_savings_usd / totals.total_baseline_cost_usd
        ELSE 0
    END AS savings_rate,
    totals.tokens_total,
    totals.tokens_to_compliance,
    COALESCE(totals.local_tier_tokens / NULLIF(totals.tiered_tokens, 0), 0)::float
        AS local_token_pct,
    COALESCE(totals.latest_projection_updated_at, NOW()) AS captured_at,
    COALESCE(model_rows.rows, '[]'::jsonb) AS rows,
    COALESCE(recent_runs.rows, '[]'::jsonb) AS recent_runs,
    totals.measured_run_count,
    totals.zero_token_run_count,
    COALESCE(warnings.rows, '[]'::jsonb) AS warnings,
    (totals.run_count > 0) AS provisioned,
    totals.latest_projection_updated_at,
    totals.tenant_id
FROM totals
LEFT JOIN model_rows ON model_rows.tenant_id = totals.tenant_id
LEFT JOIN recent_runs ON recent_runs.tenant_id = totals.tenant_id
LEFT JOIN warnings ON warnings.tenant_id = totals.tenant_id;

ALTER VIEW public.projection_cost_savings_overview SET (security_invoker = true);
