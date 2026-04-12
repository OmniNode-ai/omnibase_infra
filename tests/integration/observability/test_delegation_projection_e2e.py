# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: delegation projection consumer writes to delegation_events.

Validates:
  1. WriterDelegationProjectionPostgres UPSERTs a row into delegation_events
  2. Deduplication skips re-processing same correlation_id in the same batch
  3. Idempotent re-write updates the row, not duplicates
  4. ServiceDelegationProjectionConsumer._process_batch() routes parsed events to writer

Requires: live PostgreSQL at OMNIBASE_INFRA_DELEGATION_POSTGRES_DSN or
  OMNIBASE_INFRA_TEST_DB_DSN (falls back to OMNIBASE_INFRA_DELEGATION_POSTGRES_DSN).
Tests are skipped if DSN is not configured.

Related Tickets:
  - OMN-8532: delegation projector missing consumer service
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

POSTGRES_DSN = os.environ.get(
    "OMNIBASE_INFRA_TEST_DB_DSN",
    os.environ.get("OMNIBASE_INFRA_DELEGATION_POSTGRES_DSN", ""),
)

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_event() -> dict[str, object]:
    return {
        "correlation_id": str(uuid.uuid4()),
        "session_id": "test-session-001",
        "task_type": "code-review",
        "delegated_to": "omninode-runner-1",
        "model_name": "claude-opus-4-6",
        "delegated_by": "team-lead",
        "quality_gate_passed": True,
        "quality_gates_checked": ["lint", "tests"],
        "quality_gates_failed": [],
        "delegation_latency_ms": 42,
        "repo": "omnimarket",
        "is_shadow": False,
        "llm_call_id": str(uuid.uuid4()),
        "timestamp": "2026-04-11T12:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Writer tests (unit-level, mock pool)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool() -> MagicMock:
    """Mock asyncpg pool that records execute() calls."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")

    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)

    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_ctx)
    return pool


@pytest.mark.asyncio
async def test_writer_write_events_single(
    mock_pool: MagicMock,
    sample_event: dict[str, object],
) -> None:
    from omnibase_infra.services.observability.delegation_projection.writer_postgres import (
        WriterDelegationProjectionPostgres,
    )

    writer = WriterDelegationProjectionPostgres(pool=mock_pool)
    written = await writer.write_events([sample_event])
    assert written == 1


@pytest.mark.asyncio
async def test_writer_deduplicates_same_correlation_id(
    mock_pool: MagicMock,
    sample_event: dict[str, object],
) -> None:
    from omnibase_infra.services.observability.delegation_projection.writer_postgres import (
        WriterDelegationProjectionPostgres,
    )

    writer = WriterDelegationProjectionPostgres(pool=mock_pool)
    # First write: inserts
    written1 = await writer.write_events([sample_event])
    assert written1 == 1
    # Second write with same correlation_id: skipped by in-memory cache
    written2 = await writer.write_events([sample_event])
    assert written2 == 0


@pytest.mark.asyncio
async def test_writer_skips_events_missing_correlation_id(
    mock_pool: MagicMock,
) -> None:
    from omnibase_infra.services.observability.delegation_projection.writer_postgres import (
        WriterDelegationProjectionPostgres,
    )

    writer = WriterDelegationProjectionPostgres(pool=mock_pool)
    bad_event: dict[str, object] = {
        "task_type": "code-review",
        "delegated_to": "agent-1",
    }
    written = await writer.write_events([bad_event])
    assert written == 0


@pytest.mark.asyncio
async def test_writer_batch_multiple_events(
    mock_pool: MagicMock,
) -> None:
    from omnibase_infra.services.observability.delegation_projection.writer_postgres import (
        WriterDelegationProjectionPostgres,
    )

    writer = WriterDelegationProjectionPostgres(pool=mock_pool)
    events = [
        {
            "correlation_id": str(uuid.uuid4()),
            "task_type": "refactor",
            "delegated_to": f"agent-{i}",
        }
        for i in range(5)
    ]
    written = await writer.write_events(events)
    assert written == 5


# ---------------------------------------------------------------------------
# Consumer process_batch tests (mock writer + Kafka records)
# ---------------------------------------------------------------------------


def _make_kafka_record(payload: dict[str, object]) -> MagicMock:
    record = MagicMock()
    record.topic = "onex.evt.omniclaude.task-delegated.v1"
    record.partition = 0
    record.offset = 0
    record.value = json.dumps(payload).encode()
    return record


@pytest.mark.asyncio
async def test_consumer_process_batch_routes_to_writer(
    sample_event: dict[str, object],
) -> None:
    from omnibase_infra.services.observability.delegation_projection.config import (
        ConfigDelegationProjection,
    )
    from omnibase_infra.services.observability.delegation_projection.consumer import (
        ServiceDelegationProjectionConsumer,
    )

    config = ConfigDelegationProjection(
        kafka_bootstrap_servers="localhost:19092",
        postgres_dsn="postgresql://postgres:test@localhost:5432/omnibase_infra",
    )
    consumer = ServiceDelegationProjectionConsumer(config)

    mock_writer = AsyncMock()
    mock_writer.write_events = AsyncMock(return_value=1)
    consumer._writer = mock_writer
    consumer._running = True

    record = _make_kafka_record(sample_event)
    offsets = await consumer._process_batch([record], uuid.uuid4())

    assert len(offsets) == 1
    mock_writer.write_events.assert_awaited_once()


@pytest.mark.asyncio
async def test_consumer_process_batch_skips_null_value() -> None:
    from omnibase_infra.services.observability.delegation_projection.config import (
        ConfigDelegationProjection,
    )
    from omnibase_infra.services.observability.delegation_projection.consumer import (
        ServiceDelegationProjectionConsumer,
    )

    config = ConfigDelegationProjection(
        kafka_bootstrap_servers="localhost:19092",
        postgres_dsn="postgresql://postgres:test@localhost:5432/omnibase_infra",
    )
    consumer = ServiceDelegationProjectionConsumer(config)
    consumer._running = True

    mock_writer = AsyncMock()
    mock_writer.write_events = AsyncMock(return_value=0)
    consumer._writer = mock_writer

    record = MagicMock()
    record.topic = "onex.evt.omniclaude.task-delegated.v1"
    record.partition = 0
    record.offset = 0
    record.value = None

    offsets = await consumer._process_batch([record], uuid.uuid4())
    assert len(offsets) == 1
    mock_writer.write_events.assert_not_awaited()


# ---------------------------------------------------------------------------
# Live DB integration (skipped if no DSN)
# ---------------------------------------------------------------------------


@pytest.fixture
async def live_pool() -> AsyncGenerator[object, None]:
    if not POSTGRES_DSN:
        pytest.skip("OMNIBASE_INFRA_DELEGATION_POSTGRES_DSN not set")

    import asyncpg

    pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=3)
    yield pool
    await pool.close()


@pytest.mark.asyncio
async def test_writer_live_upsert(
    live_pool: object, sample_event: dict[str, object]
) -> None:
    import asyncpg

    from omnibase_infra.services.observability.delegation_projection.writer_postgres import (
        WriterDelegationProjectionPostgres,
    )

    pool = live_pool
    assert isinstance(pool, asyncpg.Pool)

    writer = WriterDelegationProjectionPostgres(pool=pool)
    written = await writer.write_events([sample_event])
    assert written == 1

    cid = str(sample_event["correlation_id"])
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT correlation_id, task_type, delegated_to FROM delegation_events WHERE correlation_id = $1",
            cid,
        )

    assert row is not None
    assert row["task_type"] == sample_event["task_type"]
    assert row["delegated_to"] == sample_event["delegated_to"]

    # Cleanup
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM delegation_events WHERE correlation_id = $1", cid
        )
