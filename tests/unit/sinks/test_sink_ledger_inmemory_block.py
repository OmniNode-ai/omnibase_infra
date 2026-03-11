# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for InMemoryLedgerSink BLOCK drop policy (OMN-4479)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumLedgerSinkDropPolicy
from omnibase_infra.models.ledger import ModelDbQueryRequested, ModelLedgerEventBase
from omnibase_infra.sinks import InMemoryLedgerSink


def _make_test_event(op_name: str = "test_op") -> ModelDbQueryRequested:
    """Create a test ledger event."""
    correlation_id = uuid4()
    now = datetime.now(UTC)
    return ModelDbQueryRequested(
        event_id=uuid4(),
        correlation_id=correlation_id,
        idempotency_key=ModelLedgerEventBase.build_idempotency_key(
            correlation_id, op_name, "db.query.requested"
        ),
        contract_id="test_contract",
        contract_fingerprint="abc123def456",
        operation_name=op_name,
        emitted_at=now,
        query_fingerprint=f"{op_name}:no_params",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_block_policy_does_not_raise() -> None:
    """BLOCK policy must not raise NotImplementedError."""
    sink = InMemoryLedgerSink(max_size=2, drop_policy=EnumLedgerSinkDropPolicy.BLOCK)
    result1 = await sink.emit(_make_test_event("op1"))
    result2 = await sink.emit(_make_test_event("op2"))
    assert result1 is True
    assert result2 is True
    await sink.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_block_policy_waits_then_succeeds() -> None:
    """BLOCK emit succeeds once the buffer has space."""
    sink = InMemoryLedgerSink(max_size=1, drop_policy=EnumLedgerSinkDropPolicy.BLOCK)

    # Fill the buffer
    await sink.emit(_make_test_event("fill"))

    results: list[bool] = []

    async def emit_blocking() -> None:
        result = await sink.emit(_make_test_event("blocked"))
        results.append(result)

    async def drain_soon() -> None:
        await asyncio.sleep(0.05)
        await sink.drain()

    await asyncio.gather(emit_blocking(), drain_soon())

    assert results == [True]
    await sink.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_block_policy_multiple_waiters_all_resolve() -> None:
    """Multiple blocked emitters must all eventually succeed without deadlock."""
    sink = InMemoryLedgerSink(max_size=1, drop_policy=EnumLedgerSinkDropPolicy.BLOCK)

    # Fill the buffer
    await sink.emit(_make_test_event("fill"))

    results: list[bool] = []

    async def emit_blocking(name: str) -> None:
        result = await sink.emit(_make_test_event(name))
        results.append(result)

    async def drain_twice() -> None:
        for _ in range(2):
            await asyncio.sleep(0.05)
            await sink.drain()

    await asyncio.gather(
        emit_blocking("waiter1"),
        emit_blocking("waiter2"),
        drain_twice(),
    )

    assert results == [True, True]
    await sink.close()
