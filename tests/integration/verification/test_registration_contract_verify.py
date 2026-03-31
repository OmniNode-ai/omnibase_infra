# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Integration tests for registration contract verification against live infra [OMN-7040].

These tests require:
- PostgreSQL on localhost:5436 with omnibase_infra database
- Redpanda/Kafka on localhost:19092

Run with: uv run pytest tests/integration/verification/ -v -m integration
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.verify_registration import (
    verify_registration_contract,
)


def _get_db_url() -> str:
    """Get database URL from environment."""
    url = os.environ.get("OMNIBASE_INFRA_DB_URL", "")
    if not url:
        pw = os.environ.get("POSTGRES_PASSWORD", "")
        if pw:
            url = f"postgresql://postgres:{pw}@localhost:5436/omnibase_infra"
    return url


def _get_kafka_bootstrap() -> str:
    """Get Kafka bootstrap servers from environment."""
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")


def _make_live_db_query_fn() -> Any:
    """Create a live database query function using psycopg2."""
    # ONEX_EXCLUDE: any_type - dynamic import for optional dependency
    import psycopg2  # type: ignore[import-untyped]
    import psycopg2.extras  # type: ignore[import-untyped]

    db_url = _get_db_url()
    if not db_url:
        pytest.skip("No database URL configured (set OMNIBASE_INFRA_DB_URL)")

    def db_query_fn(sql: str) -> list[dict[str, Any]]:
        conn = psycopg2.connect(db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    return db_query_fn


def _make_live_kafka_admin_fn() -> Any:
    """Create a live Kafka admin function using confluent_kafka."""
    from confluent_kafka.admin import AdminClient  # type: ignore[import-untyped]

    bootstrap = _get_kafka_bootstrap()
    admin = AdminClient({"bootstrap.servers": bootstrap})

    def kafka_admin_fn() -> set[str]:
        """Return subscribed topics for the registration orchestrator consumer group."""
        groups = admin.list_consumer_groups().result()
        # Find groups matching the registration orchestrator pattern
        subscribed: set[str] = set()
        for group in groups.valid:
            if "registration_orchestrator" in group.group_id:
                desc = admin.describe_consumer_groups([group.group_id])
                for _gid, future in desc.items():
                    result = future.result()
                    for member in result.members:
                        if member.assignment and member.assignment.topic_partitions:
                            for tp in member.assignment.topic_partitions:
                                subscribed.add(tp.topic)
        return subscribed

    return kafka_admin_fn


def _make_live_watermark_fn() -> Any:
    """Create a live watermark query function using confluent_kafka."""
    from confluent_kafka import Consumer  # type: ignore[import-untyped]

    bootstrap = _get_kafka_bootstrap()

    def watermark_fn(topic: str) -> tuple[int, int]:
        consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": "verification-probe-watermark",
                "auto.offset.reset": "earliest",
            }
        )
        try:
            low, high = consumer.get_watermark_offsets(
                # ONEX_EXCLUDE: any_type - confluent_kafka TopicPartition
                __import__("confluent_kafka").TopicPartition(topic, 0),
                timeout=5.0,
            )
            return (low, high)
        finally:
            consumer.close()

    return watermark_fn


@pytest.mark.integration
class TestRegistrationContractVerifyLive:
    """Integration tests against live PostgreSQL and Kafka."""

    def test_verify_produces_report(self) -> None:
        """Verify that the full pipeline produces a report without crashing."""
        report = verify_registration_contract(
            db_query_fn=_make_live_db_query_fn(),
            kafka_admin_fn=_make_live_kafka_admin_fn(),
            watermark_fn=_make_live_watermark_fn(),
        )
        assert report.contract_name == "node_registration_orchestrator"
        assert report.node_type == "ORCHESTRATOR_GENERIC"
        assert len(report.checks) == 4

    def test_report_has_valid_verdict(self) -> None:
        """Verify verdict is one of the valid enum values."""
        report = verify_registration_contract(
            db_query_fn=_make_live_db_query_fn(),
            kafka_admin_fn=_make_live_kafka_admin_fn(),
            watermark_fn=_make_live_watermark_fn(),
        )
        assert report.overall_verdict in (
            EnumValidationVerdict.PASS,
            EnumValidationVerdict.FAIL,
            EnumValidationVerdict.QUARANTINE,
        )

    def test_report_has_fingerprint(self) -> None:
        """Verify the report has a non-empty fingerprint."""
        report = verify_registration_contract(
            db_query_fn=_make_live_db_query_fn(),
            kafka_admin_fn=_make_live_kafka_admin_fn(),
            watermark_fn=_make_live_watermark_fn(),
        )
        assert len(report.report_fingerprint) == 64
