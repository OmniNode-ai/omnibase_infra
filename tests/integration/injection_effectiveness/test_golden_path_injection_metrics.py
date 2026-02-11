# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Golden path integration tests for injection metrics and ledger verification.

These tests verify the complete end-to-end flow:
1. Injection metrics are stored after pattern processing
2. Event ledger entry is written with correct correlation ID
3. Metric values and ledger completeness are verified

This validates the two independent write paths that share a correlation ID:
- WriterInjectionEffectivenessPostgres → injection_effectiveness + pattern_hit_rates
- LedgerSinkInjectionEffectivenessPostgres → event_ledger

Related Tickets:
    - OMN-2170: Golden path: injection metrics + ledger verification
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest

# Explicitly mark all tests in this module as postgres-dependent.
# NOTE: pytestmark in conftest.py does NOT propagate to other files.
pytestmark = [pytest.mark.postgres]

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.services.observability.injection_effectiveness.ledger_sink_postgres import (
        LedgerSinkInjectionEffectivenessPostgres,
    )
    from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
        WriterInjectionEffectivenessPostgres,
    )


class TestGoldenPathInjectionMetrics:
    """Golden path: verify injection metrics storage after pattern processing."""

    @pytest.mark.asyncio
    async def test_context_utilization_writes_injection_effectiveness(
        self,
        metrics_writer: WriterInjectionEffectivenessPostgres,
        postgres_pool: asyncpg.Pool,
        cleanup_injection_test_data: dict[str, list[UUID]],
        make_context_utilization_event: Callable[
            ..., tuple[dict[str, Any], UUID, UUID]
        ],
    ) -> None:
        """Injection metrics are stored in injection_effectiveness table.

        Verifies:
        - Row is created with correct session_id and correlation_id
        - Utilization metrics (score, method, token counts) are persisted
        - Cohort assignment is stored correctly
        """
        from omnibase_infra.services.observability.injection_effectiveness.models.model_context_utilization import (
            ModelContextUtilizationEvent,
        )

        session_id = uuid4()
        correlation_id = uuid4()
        defaults, pid1, pid2 = make_context_utilization_event(
            session_id=session_id,
            correlation_id=correlation_id,
        )
        event = ModelContextUtilizationEvent(**defaults)

        # Track for cleanup
        cleanup_injection_test_data["session_ids"].append(session_id)
        cleanup_injection_test_data["pattern_ids"].extend([pid1, pid2])

        # Write metrics
        count = await metrics_writer.write_context_utilization(
            events=[event],
            correlation_id=correlation_id,
        )
        assert count == 1

        # Verify injection_effectiveness row
        async with postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    session_id, correlation_id, cohort, cohort_identity_type,
                    total_injected_tokens, patterns_injected,
                    utilization_score, utilization_method,
                    injected_identifiers_count, reused_identifiers_count,
                    created_at, updated_at
                FROM injection_effectiveness
                WHERE session_id = $1
                """,
                session_id,
            )

        assert row is not None, "injection_effectiveness row should exist"
        assert row["session_id"] == session_id
        assert row["correlation_id"] == correlation_id
        assert row["cohort"] == "treatment"
        assert row["cohort_identity_type"] == "session_id"
        assert row["total_injected_tokens"] == 1500
        assert row["patterns_injected"] == 2
        assert abs(row["utilization_score"] - 0.85) < 0.001
        assert row["utilization_method"] == "identifier_match"
        assert row["injected_identifiers_count"] == 20
        assert row["reused_identifiers_count"] == 17
        assert row["created_at"] is not None
        assert row["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_context_utilization_writes_pattern_hit_rates(
        self,
        metrics_writer: WriterInjectionEffectivenessPostgres,
        postgres_pool: asyncpg.Pool,
        cleanup_injection_test_data: dict[str, list[UUID]],
        make_context_utilization_event: Callable[
            ..., tuple[dict[str, Any], UUID, UUID]
        ],
    ) -> None:
        """Pattern hit rates are recorded for each injected pattern.

        Verifies:
        - One row per pattern_id in pattern_hit_rates
        - Utilization scores are persisted
        - Hit/miss classification is correct (threshold = 0.5 default)
        - sample_count starts at 1 for new patterns
        """
        from omnibase_infra.services.observability.injection_effectiveness.models.model_context_utilization import (
            ModelContextUtilizationEvent,
        )

        session_id = uuid4()
        correlation_id = uuid4()
        defaults, pid1, pid2 = make_context_utilization_event(
            session_id=session_id,
            correlation_id=correlation_id,
        )
        event = ModelContextUtilizationEvent(**defaults)

        cleanup_injection_test_data["session_ids"].append(session_id)
        cleanup_injection_test_data["pattern_ids"].extend([pid1, pid2])

        await metrics_writer.write_context_utilization(
            events=[event],
            correlation_id=correlation_id,
        )

        # Verify pattern_hit_rates rows
        async with postgres_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    pattern_id, utilization_method, utilization_score,
                    hit_count, miss_count, sample_count, confidence
                FROM pattern_hit_rates
                WHERE pattern_id = ANY($1::uuid[])
                ORDER BY utilization_score DESC
                """,
                [str(pid1), str(pid2)],
            )

        assert len(rows) == 2, f"Expected 2 pattern rows, got {len(rows)}"

        # Pattern 1 (score 0.9 > 0.5 threshold → hit)
        high_score_row = next(r for r in rows if r["pattern_id"] == pid1)
        assert abs(high_score_row["utilization_score"] - 0.9) < 0.001
        assert high_score_row["hit_count"] == 1
        assert high_score_row["miss_count"] == 0
        assert high_score_row["sample_count"] == 1
        assert high_score_row["confidence"] is None  # sample_count < 20

        # Pattern 2 (score 0.8 > 0.5 threshold → hit)
        mid_score_row = next(r for r in rows if r["pattern_id"] == pid2)
        assert abs(mid_score_row["utilization_score"] - 0.8) < 0.001
        assert mid_score_row["hit_count"] == 1
        assert mid_score_row["miss_count"] == 0
        assert mid_score_row["sample_count"] == 1
        assert mid_score_row["confidence"] is None

    @pytest.mark.asyncio
    async def test_upsert_idempotency_preserves_latest_metrics(
        self,
        metrics_writer: WriterInjectionEffectivenessPostgres,
        postgres_pool: asyncpg.Pool,
        cleanup_injection_test_data: dict[str, list[UUID]],
        make_context_utilization_event: Callable[
            ..., tuple[dict[str, Any], UUID, UUID]
        ],
    ) -> None:
        """Writing the same session_id twice updates metrics (upsert semantics).

        Verifies:
        - Second write updates utilization_score, not duplicates
        - updated_at timestamp advances
        """
        from omnibase_infra.services.observability.injection_effectiveness.models.model_context_utilization import (
            ModelContextUtilizationEvent,
        )

        session_id = uuid4()
        correlation_id = uuid4()

        # First write with score 0.6
        defaults1, pid1a, pid2a = make_context_utilization_event(
            session_id=session_id,
            correlation_id=correlation_id,
            utilization_score=0.6,
        )
        event1 = ModelContextUtilizationEvent(**defaults1)
        cleanup_injection_test_data["session_ids"].append(session_id)
        cleanup_injection_test_data["pattern_ids"].extend([pid1a, pid2a])

        await metrics_writer.write_context_utilization(
            events=[event1],
            correlation_id=correlation_id,
        )

        # Capture initial updated_at
        async with postgres_pool.acquire() as conn:
            initial_row = await conn.fetchrow(
                "SELECT utilization_score, updated_at FROM injection_effectiveness WHERE session_id = $1",
                session_id,
            )
        assert initial_row is not None
        assert abs(initial_row["utilization_score"] - 0.6) < 0.001

        # Second write with score 0.95 (same session_id → upsert)
        defaults2, pid1b, pid2b = make_context_utilization_event(
            session_id=session_id,
            correlation_id=correlation_id,
            utilization_score=0.95,
        )
        event2 = ModelContextUtilizationEvent(**defaults2)
        cleanup_injection_test_data["pattern_ids"].extend([pid1b, pid2b])

        await metrics_writer.write_context_utilization(
            events=[event2],
            correlation_id=correlation_id,
        )

        async with postgres_pool.acquire() as conn:
            updated_row = await conn.fetchrow(
                "SELECT utilization_score, updated_at FROM injection_effectiveness WHERE session_id = $1",
                session_id,
            )

        assert updated_row is not None
        assert abs(updated_row["utilization_score"] - 0.95) < 0.001
        assert updated_row["updated_at"] >= initial_row["updated_at"]


class TestGoldenPathLedgerVerification:
    """Golden path: verify event ledger entries with correct correlation IDs."""

    @pytest.mark.asyncio
    async def test_ledger_entry_written_with_correct_correlation_id(
        self,
        ledger_sink: LedgerSinkInjectionEffectivenessPostgres,
        postgres_pool: asyncpg.Pool,
        cleanup_injection_test_data: dict[str, list[UUID]],
    ) -> None:
        """Event ledger entry is written with traceable correlation ID.

        Verifies:
        - Ledger entry is created successfully (non-None UUID returned)
        - correlation_id matches the input exactly
        - event_type and source are stored correctly
        - onex_headers contain session context
        """
        session_id = uuid4()
        correlation_id = uuid4()
        kafka_offset = int(uuid4().int % (2**62))

        event_payload = json.dumps(
            {
                "session_id": str(session_id),
                "utilization_score": 0.85,
                "patterns_injected": 2,
            }
        ).encode("utf-8")

        ledger_entry_id = await ledger_sink.append_session_entry(
            session_id=session_id,
            correlation_id=correlation_id,
            event_type="context_utilization",
            event_payload=event_payload,
            kafka_topic="onex.evt.omniclaude.context-utilization.v1",
            kafka_partition=0,
            kafka_offset=kafka_offset,
        )

        assert ledger_entry_id is not None, "Ledger entry should be created"
        cleanup_injection_test_data["ledger_entry_ids"].append(ledger_entry_id)

        # Verify in database
        async with postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    ledger_entry_id, topic, partition, kafka_offset,
                    event_key, event_value, correlation_id,
                    event_type, source, onex_headers, ledger_written_at
                FROM event_ledger
                WHERE ledger_entry_id = $1
                """,
                ledger_entry_id,
            )

        assert row is not None, "Ledger row should exist in database"
        assert row["correlation_id"] == correlation_id
        assert row["event_type"] == "context_utilization"
        assert row["source"] == "injection-effectiveness-consumer"
        assert row["topic"] == "onex.evt.omniclaude.context-utilization.v1"
        assert row["partition"] == 0
        assert row["kafka_offset"] == kafka_offset
        assert row["ledger_written_at"] is not None

        # Verify onex_headers contain session context
        headers = json.loads(row["onex_headers"])
        assert headers["session_id"] == str(session_id)
        assert headers["source"] == "injection-effectiveness-consumer"
        assert headers["event_type"] == "context_utilization"

        # Verify event_value roundtrip
        decoded_payload = json.loads(row["event_value"].decode("utf-8"))
        assert decoded_payload["session_id"] == str(session_id)
        assert decoded_payload["utilization_score"] == 0.85

    @pytest.mark.asyncio
    async def test_ledger_idempotent_duplicate_skip(
        self,
        ledger_sink: LedgerSinkInjectionEffectivenessPostgres,
        cleanup_injection_test_data: dict[str, list[UUID]],
    ) -> None:
        """Duplicate ledger writes are idempotently skipped.

        Verifies:
        - First write returns a ledger_entry_id (UUID)
        - Second write with same (topic, partition, offset) returns None
        """
        session_id = uuid4()
        correlation_id = uuid4()
        kafka_offset = int(uuid4().int % (2**62))

        event_payload = json.dumps({"test": "idempotency"}).encode("utf-8")

        # First write succeeds
        first_id = await ledger_sink.append_session_entry(
            session_id=session_id,
            correlation_id=correlation_id,
            event_type="context_utilization",
            event_payload=event_payload,
            kafka_topic="onex.evt.omniclaude.context-utilization.v1",
            kafka_partition=0,
            kafka_offset=kafka_offset,
        )
        assert first_id is not None
        cleanup_injection_test_data["ledger_entry_ids"].append(first_id)

        # Second write is idempotently skipped (same topic/partition/offset)
        second_id = await ledger_sink.append_session_entry(
            session_id=session_id,
            correlation_id=correlation_id,
            event_type="context_utilization",
            event_payload=event_payload,
            kafka_topic="onex.evt.omniclaude.context-utilization.v1",
            kafka_partition=0,
            kafka_offset=kafka_offset,
        )
        assert second_id is None, (
            "Duplicate should return None (ON CONFLICT DO NOTHING)"
        )

    @pytest.mark.asyncio
    async def test_ledger_batch_append(
        self,
        ledger_sink: LedgerSinkInjectionEffectivenessPostgres,
        postgres_pool: asyncpg.Pool,
        cleanup_injection_test_data: dict[str, list[UUID]],
    ) -> None:
        """Batch ledger append writes multiple entries atomically.

        Verifies:
        - All entries in batch are written
        - Each entry has correct correlation_id
        - Entries are queryable by correlation_id
        """
        correlation_id = uuid4()
        session_ids = [uuid4() for _ in range(3)]
        base_offset = int(uuid4().int % (2**60))

        from omnibase_infra.services.observability.injection_effectiveness.ledger_sink_postgres import (
            LedgerEntryDict,
        )

        entries: list[LedgerEntryDict] = [
            LedgerEntryDict(
                session_id=sid,
                event_type="context_utilization",
                event_payload=json.dumps({"session_id": str(sid), "index": i}).encode(
                    "utf-8"
                ),
                kafka_topic="onex.evt.omniclaude.context-utilization.v1",
                kafka_partition=0,
                kafka_offset=base_offset + i,
            )
            for i, sid in enumerate(session_ids)
        ]

        count = await ledger_sink.append_batch(
            entries=entries,
            correlation_id=correlation_id,
        )
        assert count == 3

        # Query by correlation_id to verify all entries
        async with postgres_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ledger_entry_id, correlation_id, event_type, source
                FROM event_ledger
                WHERE correlation_id = $1
                ORDER BY kafka_offset
                """,
                correlation_id,
            )

        # Track for cleanup
        for row in rows:
            cleanup_injection_test_data["ledger_entry_ids"].append(
                row["ledger_entry_id"]
            )

        assert len(rows) == 3, f"Expected 3 ledger entries, got {len(rows)}"
        for row in rows:
            assert row["correlation_id"] == correlation_id
            assert row["event_type"] == "context_utilization"
            assert row["source"] == "injection-effectiveness-consumer"


class TestGoldenPathEndToEnd:
    """Golden path: full end-to-end injection metrics + ledger in single flow."""

    @pytest.mark.asyncio
    async def test_metrics_and_ledger_share_correlation_id(
        self,
        metrics_writer: WriterInjectionEffectivenessPostgres,
        ledger_sink: LedgerSinkInjectionEffectivenessPostgres,
        postgres_pool: asyncpg.Pool,
        cleanup_injection_test_data: dict[str, list[UUID]],
        make_context_utilization_event: Callable[
            ..., tuple[dict[str, Any], UUID, UUID]
        ],
    ) -> None:
        """Full golden path: metrics + ledger entries share traceable correlation ID.

        This test simulates the complete injection effectiveness pipeline:
        1. Create a context utilization event
        2. Write injection metrics (writer)
        3. Write audit ledger entry (sink)
        4. Verify both writes share the same correlation_id
        5. Verify metric values and ledger completeness

        This is the core acceptance criteria for OMN-2170.
        """
        from omnibase_infra.services.observability.injection_effectiveness.models.model_context_utilization import (
            ModelContextUtilizationEvent,
        )

        session_id = uuid4()
        correlation_id = uuid4()
        kafka_offset = int(uuid4().int % (2**62))

        defaults, pid1, pid2 = make_context_utilization_event(
            session_id=session_id,
            correlation_id=correlation_id,
            utilization_score=0.78,
            total_injected_tokens=2000,
            patterns_injected=2,
        )
        event = ModelContextUtilizationEvent(**defaults)

        # Track for cleanup
        cleanup_injection_test_data["session_ids"].append(session_id)
        cleanup_injection_test_data["pattern_ids"].extend([pid1, pid2])

        # Step 1: Write injection metrics
        metrics_count = await metrics_writer.write_context_utilization(
            events=[event],
            correlation_id=correlation_id,
        )
        assert metrics_count == 1

        # Step 2: Write ledger entry (simulating what the consumer does)
        event_payload = json.dumps(
            {
                "session_id": str(session_id),
                "correlation_id": str(correlation_id),
                "utilization_score": 0.78,
                "total_injected_tokens": 2000,
                "patterns_injected": 2,
            }
        ).encode("utf-8")

        ledger_entry_id = await ledger_sink.append_session_entry(
            session_id=session_id,
            correlation_id=correlation_id,
            event_type="context_utilization",
            event_payload=event_payload,
            kafka_topic="onex.evt.omniclaude.context-utilization.v1",
            kafka_partition=0,
            kafka_offset=kafka_offset,
        )
        assert ledger_entry_id is not None
        cleanup_injection_test_data["ledger_entry_ids"].append(ledger_entry_id)

        # Step 3: Verify both writes share the same correlation_id
        async with postgres_pool.acquire() as conn:
            # Check injection_effectiveness
            metrics_row = await conn.fetchrow(
                "SELECT correlation_id, utilization_score, total_injected_tokens, patterns_injected "
                "FROM injection_effectiveness WHERE session_id = $1",
                session_id,
            )

            # Check event_ledger
            ledger_row = await conn.fetchrow(
                "SELECT correlation_id, event_type, source "
                "FROM event_ledger WHERE ledger_entry_id = $1",
                ledger_entry_id,
            )

            # Check pattern_hit_rates
            pattern_rows = await conn.fetch(
                "SELECT pattern_id, utilization_score, hit_count, miss_count, sample_count "
                "FROM pattern_hit_rates WHERE pattern_id = ANY($1::uuid[])",
                [str(pid1), str(pid2)],
            )

        # Verify metrics row
        assert metrics_row is not None, "Metrics row must exist"
        assert metrics_row["correlation_id"] == correlation_id
        assert abs(metrics_row["utilization_score"] - 0.78) < 0.001
        assert metrics_row["total_injected_tokens"] == 2000
        assert metrics_row["patterns_injected"] == 2

        # Verify ledger row
        assert ledger_row is not None, "Ledger row must exist"
        assert ledger_row["correlation_id"] == correlation_id
        assert ledger_row["event_type"] == "context_utilization"
        assert ledger_row["source"] == "injection-effectiveness-consumer"

        # Verify pattern hit rates
        assert len(pattern_rows) == 2, "Both patterns should have hit rate entries"

        # THE GOLDEN ASSERTION: same correlation_id links metrics to ledger
        assert metrics_row["correlation_id"] == ledger_row["correlation_id"], (
            "Metrics and ledger must share the same correlation_id for traceability"
        )

    @pytest.mark.asyncio
    async def test_ledger_completeness_all_fields_populated(
        self,
        ledger_sink: LedgerSinkInjectionEffectivenessPostgres,
        postgres_pool: asyncpg.Pool,
        cleanup_injection_test_data: dict[str, list[UUID]],
    ) -> None:
        """Ledger entry has all required fields populated (completeness check).

        Verifies every column in event_ledger is non-null for a well-formed entry.
        This guards against silent schema regressions where fields become nullable
        or default values change unexpectedly.
        """
        session_id = uuid4()
        correlation_id = uuid4()
        kafka_offset = int(uuid4().int % (2**62))
        event_timestamp = datetime.now(UTC)

        event_payload = json.dumps(
            {
                "session_id": str(session_id),
                "completeness": "test",
            }
        ).encode("utf-8")

        ledger_entry_id = await ledger_sink.append_session_entry(
            session_id=session_id,
            correlation_id=correlation_id,
            event_type="context_utilization",
            event_payload=event_payload,
            kafka_topic="onex.evt.omniclaude.context-utilization.v1",
            kafka_partition=0,
            kafka_offset=kafka_offset,
            event_timestamp=event_timestamp,
        )
        assert ledger_entry_id is not None
        cleanup_injection_test_data["ledger_entry_ids"].append(ledger_entry_id)

        async with postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    ledger_entry_id, topic, partition, kafka_offset,
                    event_key, event_value, correlation_id,
                    event_type, source, onex_headers,
                    event_timestamp, ledger_written_at
                FROM event_ledger
                WHERE ledger_entry_id = $1
                """,
                ledger_entry_id,
            )

        assert row is not None
        # Verify all fields are populated (completeness)
        assert row["ledger_entry_id"] is not None, "ledger_entry_id must be populated"
        assert row["topic"] is not None, "topic must be populated"
        assert row["partition"] is not None, "partition must be populated"
        assert row["kafka_offset"] is not None, "kafka_offset must be populated"
        assert row["event_key"] is not None, "event_key must be populated"
        assert row["event_value"] is not None, "event_value must be populated"
        assert row["correlation_id"] is not None, "correlation_id must be populated"
        assert row["event_type"] is not None, "event_type must be populated"
        assert row["source"] is not None, "source must be populated"
        assert row["onex_headers"] is not None, "onex_headers must be populated"
        assert row["event_timestamp"] is not None, "event_timestamp must be populated"
        assert row["ledger_written_at"] is not None, (
            "ledger_written_at must be populated"
        )
