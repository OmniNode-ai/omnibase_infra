# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for ServiceFreshnessMonitor (OMN-11200).

Validates the monitor wires correctly with:
- Projection contract registry contracts
- Topic registry resolution
- Event bus envelope publishing
- Start/stop lifecycle with the async loop
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.models.health.model_projection_degraded_event import (
    ModelProjectionDegradedEvent,
)
from omnibase_infra.models.health.model_projection_recovered_event import (
    ModelProjectionRecoveredEvent,
)
from omnibase_infra.models.projection.projection_contract_registry import (
    PROJECTION_CONTRACTS,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.freshness_monitor import ServiceFreshnessMonitor


def _make_stub_registry() -> MagicMock:
    registry = MagicMock()
    registry.resolve.side_effect = lambda key: (
        "onex.evt.omnibase-infra.projection-freshness-degraded.v1"
        if "DEGRADED" in key
        else "onex.evt.omnibase-infra.projection-freshness-recovered.v1"
    )
    return registry


@pytest.mark.integration
class TestFreshnessMonitorIntegration:
    @pytest.mark.asyncio
    async def test_registry_contracts_produce_degraded_events(self) -> None:
        """All PROJECTION_CONTRACTS produce degraded events when stale."""
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=120)

        async def query(table: str, field: str) -> datetime:
            return stale_ts

        bus = MagicMock(spec=ProtocolEventBusLike)
        bus.publish_envelope = AsyncMock()

        monitor = ServiceFreshnessMonitor(
            contracts=PROJECTION_CONTRACTS,
            query_fn=query,
            event_bus=bus,
            check_interval_seconds=60.0,
            topic_registry=_make_stub_registry(),
        )
        events = await monitor.run_once()

        assert len(events) == len(PROJECTION_CONTRACTS)
        for event in events:
            assert isinstance(event, ModelProjectionDegradedEvent)
        assert bus.publish_envelope.await_count == len(PROJECTION_CONTRACTS)

    @pytest.mark.asyncio
    async def test_full_degrade_recover_cycle_with_bus(self) -> None:
        """Full degradation → recovery cycle emits correct envelopes."""
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=120)
        fresh_ts = now - timedelta(seconds=5)
        contracts = PROJECTION_CONTRACTS[:1]
        call_count = 0

        async def query(table: str, field: str) -> datetime:
            nonlocal call_count
            call_count += 1
            return stale_ts if call_count <= 1 else fresh_ts

        bus = MagicMock(spec=ProtocolEventBusLike)
        bus.publish_envelope = AsyncMock()

        monitor = ServiceFreshnessMonitor(
            contracts=contracts,
            query_fn=query,
            event_bus=bus,
            check_interval_seconds=60.0,
            topic_registry=_make_stub_registry(),
        )

        degraded = await monitor.run_once()
        recovered = await monitor.run_once()

        assert len(degraded) == 1
        assert isinstance(degraded[0], ModelProjectionDegradedEvent)
        assert len(recovered) == 1
        assert isinstance(recovered[0], ModelProjectionRecoveredEvent)

        assert bus.publish_envelope.await_count == 2
        degraded_topic = bus.publish_envelope.call_args_list[0].kwargs["topic"]
        recovered_topic = bus.publish_envelope.call_args_list[1].kwargs["topic"]
        assert "degraded" in degraded_topic
        assert "recovered" in recovered_topic

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self) -> None:
        """Monitor starts and stops cleanly without hanging."""

        async def query(table: str, field: str) -> datetime:
            return datetime.now(UTC)

        monitor = ServiceFreshnessMonitor(
            contracts=PROJECTION_CONTRACTS,
            query_fn=query,
            check_interval_seconds=60.0,
            topic_registry=_make_stub_registry(),
        )

        await monitor.start()
        assert monitor._running is True
        assert monitor._task is not None

        await monitor.stop()
        assert monitor._running is False
        assert monitor._task is None
