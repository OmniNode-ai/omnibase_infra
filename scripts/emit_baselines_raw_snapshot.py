#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Minimal baselines raw measurement emitter.

Emits a valid baselines-computed snapshot event to Kafka with the payload
shape expected by the omnidash ReadModelConsumer's projectBaselinesSnapshot().

This is a RAW MEASUREMENTS emitter -- it does NOT compute deltas, ROI, or
promotion recommendations. It reads whatever pattern execution data exists
in the omnibase_infra database (from agent_routing_decisions + agent_actions)
and packages it into the consumer-expected format.

If no real pattern execution data is available, emits an honest empty snapshot
(valid schema, zero comparisons) so the pipeline is verified end-to-end.

Payload shape (consumed by omnidash projectBaselinesSnapshot):
    - snapshot_id: UUID
    - contract_version: int (1)
    - computed_at_utc: ISO datetime
    - window_start_utc: ISO datetime | null
    - window_end_utc: ISO datetime | null
    - comparisons: list of comparison objects with DeltaMetric JSONB fields
    - trend: list of {date, avg_cost_savings, avg_outcome_improvement, comparisons_evaluated}
    - breakdown: list of {action, count, avg_confidence}

Usage:
    uv run python scripts/emit_baselines_raw_snapshot.py

    # Dry-run mode (prints payload to stdout, no Kafka):
    uv run python scripts/emit_baselines_raw_snapshot.py --dry-run

    # With DB (reads real data if available):
    OMNIBASE_INFRA_DB_URL=postgresql://... uv run python scripts/emit_baselines_raw_snapshot.py

Environment Variables:
    OMNIBASE_INFRA_DB_URL (optional)
        Full PostgreSQL DSN. When set, reads real pattern execution data.
        When unset, emits an empty snapshot for pipeline verification.

    KAFKA_BOOTSTRAP_SERVERS (optional, default: localhost:19092)
        Kafka bootstrap address. Set to empty string to skip emission.

Exit Codes:
    0  Success
    1  Configuration or runtime error

Ticket: SOW-Phase2 B4a
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("emit_baselines_raw_snapshot")

_TOPIC_BASELINES_COMPUTED = "onex.evt.omnibase-infra.baselines-computed.v1"


def _make_delta_metric(
    *,
    label: str,
    baseline: float,
    candidate: float,
    unit: str = "",
) -> dict[str, Any]:
    """Build a DeltaMetric JSONB object matching the omnidash consumer schema.

    DeltaMetric shape: { label, baseline, candidate, delta, direction, unit }
    direction: 'better' if candidate < baseline (for costs) or candidate > baseline (for rates)
    For raw measurements, we report delta = baseline - candidate (positive = savings).
    """
    delta = baseline - candidate
    # For raw measurements, positive delta means candidate used fewer resources
    if abs(delta) < 1e-9:
        direction = "neutral"
    elif delta > 0:
        direction = "better"
    else:
        direction = "worse"

    return {
        "label": label,
        "baseline": round(baseline, 4),
        "candidate": round(candidate, 4),
        "delta": round(delta, 4),
        "direction": direction,
        "unit": unit,
    }


def _build_empty_snapshot() -> dict[str, Any]:
    """Build a valid empty snapshot for pipeline verification.

    This is an honest empty snapshot -- zero comparisons, zero trend points,
    zero breakdown rows. The omnidash consumer will accept it and upsert
    an empty baselines_snapshots row, proving the pipeline works end-to-end.
    """
    now = datetime.now(tz=UTC)
    return {
        "snapshot_id": str(uuid4()),
        "contract_version": 1,
        "computed_at_utc": now.isoformat(),
        "window_start_utc": None,
        "window_end_utc": None,
        "comparisons": [],
        "trend": [],
        "breakdown": [],
    }


async def _read_raw_measurements(db_url: str) -> dict[str, Any]:
    """Read pattern execution data from omnibase_infra DB and package as snapshot.

    Queries agent_routing_decisions + agent_actions to build raw measurements
    in the format expected by the omnidash consumer.

    If no data exists, returns an empty snapshot.
    """
    import asyncpg

    now = datetime.now(tz=UTC)
    window_start = now - timedelta(days=90)

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2, command_timeout=30)
    try:
        comparisons = await _read_comparison_data(pool, window_start, now)
        trend = await _read_trend_data(pool, window_start, now)
        breakdown = _compute_breakdown_from_comparisons(comparisons)

        snapshot: dict[str, Any] = {
            "snapshot_id": str(uuid4()),
            "contract_version": 1,
            "computed_at_utc": now.isoformat(),
            "window_start_utc": window_start.isoformat() if comparisons else None,
            "window_end_utc": now.isoformat() if comparisons else None,
            "comparisons": comparisons,
            "trend": trend,
            "breakdown": breakdown,
        }

        logger.info(
            "Built snapshot from DB: comparisons=%d trend=%d breakdown=%d",
            len(comparisons),
            len(trend),
            len(breakdown),
        )
        return snapshot
    finally:
        await pool.close()


async def _read_comparison_data(
    pool: Any,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    """Read per-pattern comparison data from agent_routing_decisions.

    Groups by selected_agent (as pattern proxy) and computes raw measurements
    for treatment vs control cohorts.
    """
    sql = """
        WITH agent_cohorts AS (
            SELECT
                rd.selected_agent AS pattern_label,
                md5(rd.selected_agent)::text AS pattern_id,
                CASE
                    WHEN rd.confidence_score >= 0.7 THEN 'treatment'
                    ELSE 'control'
                END AS cohort,
                rd.correlation_id,
                action_stats.avg_duration_ms,
                action_stats.total_tokens,
                action_stats.success_rate,
                action_stats.retry_count,
                action_stats.review_iterations
            FROM agent_routing_decisions rd
            LEFT JOIN LATERAL (
                SELECT
                    AVG(aa.duration_ms) AS avg_duration_ms,
                    COALESCE(SUM(aa.total_tokens), 0) AS total_tokens,
                    COALESCE(
                        CAST(COUNT(*) FILTER (WHERE aa.status = 'completed') AS FLOAT)
                        / NULLIF(COUNT(*), 0),
                        0.0
                    ) AS success_rate,
                    COALESCE(SUM(aa.retry_count), 0) AS retry_count,
                    COALESCE(MAX(aa.review_iterations), 0) AS review_iterations
                FROM agent_actions aa
                WHERE aa.correlation_id = rd.correlation_id
            ) action_stats ON TRUE
            WHERE rd.selected_agent IS NOT NULL
                AND rd.correlation_id IS NOT NULL
                AND rd.created_at >= $1
                AND rd.created_at <= $2
        ),
        pattern_agg AS (
            SELECT
                pattern_id,
                pattern_label,
                COUNT(*) AS sample_size,
                -- Treatment metrics
                AVG(total_tokens) FILTER (WHERE cohort = 'treatment') AS treatment_tokens,
                AVG(avg_duration_ms) FILTER (WHERE cohort = 'treatment') AS treatment_latency,
                AVG(retry_count) FILTER (WHERE cohort = 'treatment') AS treatment_retries,
                AVG(success_rate) FILTER (WHERE cohort = 'treatment') AS treatment_pass_rate,
                AVG(review_iterations) FILTER (WHERE cohort = 'treatment') AS treatment_review_iters,
                -- Control metrics
                AVG(total_tokens) FILTER (WHERE cohort = 'control') AS control_tokens,
                AVG(avg_duration_ms) FILTER (WHERE cohort = 'control') AS control_latency,
                AVG(retry_count) FILTER (WHERE cohort = 'control') AS control_retries,
                AVG(success_rate) FILTER (WHERE cohort = 'control') AS control_pass_rate,
                AVG(review_iterations) FILTER (WHERE cohort = 'control') AS control_review_iters
            FROM agent_cohorts
            GROUP BY pattern_id, pattern_label
            ORDER BY sample_size DESC
            LIMIT 100
        )
        SELECT * FROM pattern_agg
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, window_start, window_end)

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        # Build DeltaMetric JSONB objects for each dimension
        treatment_tokens = float(row["treatment_tokens"] or 0)
        control_tokens = float(row["control_tokens"] or 0)
        treatment_latency = float(row["treatment_latency"] or 0)
        control_latency = float(row["control_latency"] or 0)
        treatment_retries = float(row["treatment_retries"] or 0)
        control_retries = float(row["control_retries"] or 0)
        treatment_pass_rate = float(row["treatment_pass_rate"] or 0)
        control_pass_rate = float(row["control_pass_rate"] or 0)
        treatment_review_iters = float(row["treatment_review_iters"] or 0)
        control_review_iters = float(row["control_review_iters"] or 0)

        sample_size = int(row["sample_size"] or 0)

        # Determine recommendation based on raw data
        # RAW only -- no real computation, just classify based on data availability
        if sample_size < 5:
            recommendation = "shadow"
            confidence = "low"
            rationale = f"Insufficient sample size ({sample_size} < 5)"
        elif (
            treatment_pass_rate >= control_pass_rate
            and treatment_tokens <= control_tokens
        ):
            recommendation = "promote"
            confidence = "medium" if sample_size >= 20 else "low"
            rationale = (
                "Treatment shows equal or better pass rate with equal or fewer tokens"
            )
        elif treatment_pass_rate < control_pass_rate:
            recommendation = "suppress"
            confidence = "medium" if sample_size >= 20 else "low"
            rationale = "Treatment shows lower pass rate than control"
        else:
            recommendation = "shadow"
            confidence = "low"
            rationale = "Insufficient evidence for promotion or suppression"

        comparison: dict[str, Any] = {
            "pattern_id": row["pattern_id"],
            "pattern_name": row["pattern_label"] or row["pattern_id"],
            "sample_size": sample_size,
            "window_start": window_start.isoformat()
            if hasattr(window_start, "isoformat")
            else str(window_start),
            "window_end": window_end.isoformat()
            if hasattr(window_end, "isoformat")
            else str(window_end),
            "token_delta": _make_delta_metric(
                label="Token Usage",
                baseline=control_tokens,
                candidate=treatment_tokens,
                unit="tokens",
            ),
            "time_delta": _make_delta_metric(
                label="Latency",
                baseline=control_latency,
                candidate=treatment_latency,
                unit="ms",
            ),
            "retry_delta": _make_delta_metric(
                label="Retries",
                baseline=control_retries,
                candidate=treatment_retries,
                unit="retries",
            ),
            "test_pass_rate_delta": _make_delta_metric(
                label="Test Pass Rate",
                baseline=control_pass_rate,
                candidate=treatment_pass_rate,
                unit="rate",
            ),
            "review_iteration_delta": _make_delta_metric(
                label="Review Iterations",
                baseline=control_review_iters,
                candidate=treatment_review_iters,
                unit="iterations",
            ),
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale,
        }
        comparisons.append(comparison)

    return comparisons


async def _read_trend_data(
    pool: Any,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    """Read daily trend data from agent_routing_decisions.

    Produces trend rows in the format expected by omnidash:
    {date: "YYYY-MM-DD", avg_cost_savings, avg_outcome_improvement, comparisons_evaluated}
    """
    sql = """
        WITH daily_cohorts AS (
            SELECT
                DATE(rd.created_at) AS day,
                CASE
                    WHEN rd.confidence_score >= 0.7 THEN 'treatment'
                    ELSE 'control'
                END AS cohort,
                action_stats.total_tokens,
                action_stats.success_rate
            FROM agent_routing_decisions rd
            LEFT JOIN LATERAL (
                SELECT
                    COALESCE(SUM(aa.total_tokens), 0) AS total_tokens,
                    COALESCE(
                        CAST(COUNT(*) FILTER (WHERE aa.status = 'completed') AS FLOAT)
                        / NULLIF(COUNT(*), 0),
                        0.0
                    ) AS success_rate
                FROM agent_actions aa
                WHERE aa.correlation_id = rd.correlation_id
            ) action_stats ON TRUE
            WHERE rd.correlation_id IS NOT NULL
                AND rd.created_at >= $1
                AND rd.created_at <= $2
        ),
        daily_agg AS (
            SELECT
                day,
                COUNT(*) AS comparisons_evaluated,
                -- Cost savings: (control_tokens - treatment_tokens) / control_tokens
                CASE
                    WHEN AVG(total_tokens) FILTER (WHERE cohort = 'control') > 0
                    THEN (
                        AVG(total_tokens) FILTER (WHERE cohort = 'control')
                        - AVG(total_tokens) FILTER (WHERE cohort = 'treatment')
                    ) / AVG(total_tokens) FILTER (WHERE cohort = 'control')
                    ELSE 0
                END AS avg_cost_savings,
                -- Outcome improvement: treatment_pass_rate - control_pass_rate
                COALESCE(
                    AVG(success_rate) FILTER (WHERE cohort = 'treatment')
                    - AVG(success_rate) FILTER (WHERE cohort = 'control'),
                    0
                ) AS avg_outcome_improvement
            FROM daily_cohorts
            GROUP BY day
            ORDER BY day DESC
            LIMIT 90
        )
        SELECT * FROM daily_agg
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, window_start, window_end)

    trend: list[dict[str, Any]] = []
    for row in rows:
        cost_savings = float(row["avg_cost_savings"] or 0)
        outcome_improvement = float(row["avg_outcome_improvement"] or 0)
        # Clamp to [0, 99] as the omnidash consumer does
        cost_savings = max(0.0, min(cost_savings, 99.0))
        outcome_improvement = max(0.0, min(outcome_improvement, 99.0))

        trend.append(
            {
                "date": row["day"].isoformat(),
                "avg_cost_savings": round(cost_savings, 6),
                "avg_outcome_improvement": round(outcome_improvement, 6),
                "comparisons_evaluated": int(row["comparisons_evaluated"] or 0),
            }
        )

    return trend


def _compute_breakdown_from_comparisons(
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive breakdown rows from comparison recommendations.

    Groups comparisons by recommendation action and computes counts
    and average confidence.
    """
    action_counts: dict[str, list[float]] = {}
    confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}

    for comp in comparisons:
        action = comp.get("recommendation", "shadow")
        conf_label = comp.get("confidence", "low")
        conf_value = confidence_map.get(conf_label, 0.3)

        if action not in action_counts:
            action_counts[action] = []
        action_counts[action].append(conf_value)

    breakdown: list[dict[str, Any]] = []
    for action, confidences in sorted(action_counts.items()):
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        breakdown.append(
            {
                "action": action,
                "count": len(confidences),
                "avg_confidence": round(avg_conf, 4),
            }
        )

    return breakdown


async def _emit_to_kafka(
    payload: dict[str, Any],
    kafka_servers: str,
) -> None:
    """Emit the snapshot payload to Kafka."""
    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(
        bootstrap_servers=kafka_servers,
        acks="all",
        enable_idempotence=True,
    )
    try:
        await producer.start()
        logger.info("Kafka producer connected to %s", kafka_servers)

        # Wrap in ONEX envelope format (consumer unwraps payload key)
        envelope = {
            "event_type": "baselines.computed",
            "payload": payload,
            "correlation_id": payload.get("snapshot_id", str(uuid4())),
        }
        body = json.dumps(envelope, default=str).encode("utf-8")
        await producer.send_and_wait(_TOPIC_BASELINES_COMPUTED, body)
        logger.info(
            "Emitted baselines-computed snapshot to %s (snapshot_id=%s, "
            "comparisons=%d, trend=%d, breakdown=%d)",
            _TOPIC_BASELINES_COMPUTED,
            payload.get("snapshot_id"),
            len(payload.get("comparisons", [])),
            len(payload.get("trend", [])),
            len(payload.get("breakdown", [])),
        )
    finally:
        await producer.stop()


async def _run(dry_run: bool = False) -> int:
    """Main entry point."""
    db_url = os.environ.get("OMNIBASE_INFRA_DB_URL", "").strip()
    kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092").strip()

    # Build snapshot payload
    if db_url:
        logger.info("Reading raw measurements from DB")
        try:
            payload = await _read_raw_measurements(db_url)
        except Exception:
            logger.exception("Failed to read from DB, falling back to empty snapshot")
            payload = _build_empty_snapshot()
    else:
        logger.info(
            "No OMNIBASE_INFRA_DB_URL set -- emitting empty snapshot for pipeline verification"
        )
        payload = _build_empty_snapshot()

    if dry_run:
        print(json.dumps(payload, indent=2, default=str))
        logger.info("Dry-run mode -- payload printed to stdout, no Kafka emission")
        return 0

    if not kafka_servers:
        logger.warning("KAFKA_BOOTSTRAP_SERVERS is empty -- skipping emission")
        print(json.dumps(payload, indent=2, default=str))
        return 0

    try:
        await _emit_to_kafka(payload, kafka_servers)
    except Exception:
        logger.exception("Failed to emit to Kafka")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a baselines raw measurement snapshot to Kafka.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payload to stdout without emitting to Kafka.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
