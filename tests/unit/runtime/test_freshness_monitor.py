# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceFreshnessMonitor (OMN-11200).

Validates:
- Staleness detection against SLA
- Degradation event emitted with correct fields
- Recovery event emitted on resolution
- No duplicate degradation events per breach
- Contracts without freshness fields are skipped
- NULL query result is skipped without event
- Query failure is swallowed (loop continues)
- start/stop lifecycle is idempotent
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.enums.enum_degraded_behavior import EnumDegradedBehavior
from omnibase_core.models.projection.model_cursor_contract import ModelCursorContract
from omnibase_core.models.projection.model_projection_contract import (
    ModelProjectionContract,
)
from omnibase_infra.models.health.model_projection_degraded_event import (
    ModelProjectionDegradedEvent,
)
from omnibase_infra.models.health.model_projection_recovered_event import (
    ModelProjectionRecoveredEvent,
)
from omnibase_infra.runtime.freshness_monitor import ServiceFreshnessMonitor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CURSOR = ModelCursorContract(cursor_type="kafka_offset", supports_replay=True)


def _make_contract(
    name: str = "test_projection",
    sla_seconds: int = 30,
    freshness_field: str = "last_seen_at",
    freshness_source_table: str = "test_table",
    behavior: EnumDegradedBehavior = EnumDegradedBehavior.SERVE_STALE_WITH_WARNING,
) -> ModelProjectionContract:
    return ModelProjectionContract(
        projection_name=name,
        source_topics=("onex.evt.test.event.v1",),
        schema_model="omnibase_infra.models.test.ModelTest",
        freshness_sla_seconds=sla_seconds,
        freshness_field=freshness_field,
        freshness_source_table=freshness_source_table,
        degraded_semantics=behavior,
        cursor=_CURSOR,
        ordering_contract_ref=None,
    )


def _make_stub_registry(
    degraded_topic: str = "onex.evt.omnibase-infra.projection-freshness-degraded.v1",
    recovered_topic: str = "onex.evt.omnibase-infra.projection-freshness-recovered.v1",
) -> MagicMock:
    registry = MagicMock()
    registry.resolve.side_effect = lambda key: (
        degraded_topic if "DEGRADED" in key else recovered_topic
    )
    return registry


def _make_monitor(
    contracts: tuple[ModelProjectionContract, ...],
    query_fn: object,
    event_bus: object | None = None,
) -> ServiceFreshnessMonitor:
    return ServiceFreshnessMonitor(
        contracts=contracts,
        query_fn=query_fn,  # type: ignore[arg-type]
        event_bus=event_bus,
        check_interval_seconds=60.0,
        topic_registry=_make_stub_registry(),
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestFreshnessDetection:
    @pytest.mark.asyncio
    async def test_detects_staleness_correctly(self) -> None:
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=60)
        contract = _make_contract(sla_seconds=30)

        async def query(table: str, field: str) -> datetime:
            return stale_ts

        monitor = _make_monitor((contract,), query)
        events = await monitor.run_once()

        assert len(events) == 1
        assert isinstance(events[0], ModelProjectionDegradedEvent)
        assert events[0].actual_staleness_seconds > 30

    @pytest.mark.asyncio
    async def test_healthy_projection_emits_no_event(self) -> None:
        now = datetime.now(UTC)
        fresh_ts = now - timedelta(seconds=10)
        contract = _make_contract(sla_seconds=30)

        async def query(table: str, field: str) -> datetime:
            return fresh_ts

        monitor = _make_monitor((contract,), query)
        events = await monitor.run_once()

        assert events == []


# ---------------------------------------------------------------------------
# Degradation event fields
# ---------------------------------------------------------------------------


class TestDegradationEvent:
    @pytest.mark.asyncio
    async def test_degradation_event_fields(self) -> None:
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=90)
        contract = _make_contract(
            name="my_proj",
            sla_seconds=30,
            behavior=EnumDegradedBehavior.SERVE_STALE_WITH_WARNING,
        )

        async def query(table: str, field: str) -> datetime:
            return stale_ts

        monitor = _make_monitor((contract,), query)
        events = await monitor.run_once()

        assert len(events) == 1
        evt = events[0]
        assert isinstance(evt, ModelProjectionDegradedEvent)
        assert evt.projection_name == "my_proj"
        assert evt.sla_seconds == 30
        assert evt.actual_staleness_seconds > 30
        assert evt.degraded_behavior == "serve_stale_with_warning"
        assert evt.source_contract_hash  # non-empty string
        assert isinstance(evt.observed_at, datetime)

    @pytest.mark.asyncio
    async def test_degradation_event_emitted_to_bus(self) -> None:
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=60)
        contract = _make_contract(sla_seconds=30)

        async def query(table: str, field: str) -> datetime:
            return stale_ts

        bus = MagicMock()
        bus.publish_envelope = AsyncMock()
        monitor = _make_monitor((contract,), query, event_bus=bus)
        await monitor.run_once()

        bus.publish_envelope.assert_awaited_once()
        call_kwargs = bus.publish_envelope.call_args.kwargs
        assert "projection-freshness-degraded" in call_kwargs["topic"]


# ---------------------------------------------------------------------------
# Recovery event
# ---------------------------------------------------------------------------


class TestRecoveryEvent:
    @pytest.mark.asyncio
    async def test_emits_recovery_after_degradation(self) -> None:
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=60)
        fresh_ts = now - timedelta(seconds=5)
        contract = _make_contract(sla_seconds=30)
        call_count = 0

        async def query(table: str, field: str) -> datetime:
            nonlocal call_count
            call_count += 1
            return stale_ts if call_count == 1 else fresh_ts

        monitor = _make_monitor((contract,), query)

        first_events = await monitor.run_once()
        assert len(first_events) == 1
        assert isinstance(first_events[0], ModelProjectionDegradedEvent)

        second_events = await monitor.run_once()
        assert len(second_events) == 1
        assert isinstance(second_events[0], ModelProjectionRecoveredEvent)
        assert second_events[0].projection_name == contract.projection_name
        assert second_events[0].sla_seconds == 30

    @pytest.mark.asyncio
    async def test_recovery_event_emitted_to_bus(self) -> None:
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=60)
        fresh_ts = now - timedelta(seconds=5)
        contract = _make_contract(sla_seconds=30)
        call_count = 0

        async def query(table: str, field: str) -> datetime:
            nonlocal call_count
            call_count += 1
            return stale_ts if call_count == 1 else fresh_ts

        bus = MagicMock()
        bus.publish_envelope = AsyncMock()
        monitor = _make_monitor((contract,), query, event_bus=bus)

        await monitor.run_once()
        await monitor.run_once()

        assert bus.publish_envelope.await_count == 2
        recovery_call = bus.publish_envelope.call_args_list[1]
        assert "projection-freshness-recovered" in recovery_call.kwargs["topic"]


# ---------------------------------------------------------------------------
# No duplicate events
# ---------------------------------------------------------------------------


class TestNoDuplicates:
    @pytest.mark.asyncio
    async def test_no_duplicate_degradation_events(self) -> None:
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=60)
        contract = _make_contract(sla_seconds=30)

        async def query(table: str, field: str) -> datetime:
            return stale_ts

        monitor = _make_monitor((contract,), query)

        first = await monitor.run_once()
        second = await monitor.run_once()
        third = await monitor.run_once()

        assert len(first) == 1
        assert len(second) == 0
        assert len(third) == 0

    @pytest.mark.asyncio
    async def test_no_duplicate_recovery_events(self) -> None:
        now = datetime.now(UTC)
        stale_ts = now - timedelta(seconds=60)
        fresh_ts = now - timedelta(seconds=5)
        contract = _make_contract(sla_seconds=30)
        call_count = 0

        async def query(table: str, field: str) -> datetime:
            nonlocal call_count
            call_count += 1
            return stale_ts if call_count <= 1 else fresh_ts

        monitor = _make_monitor((contract,), query)

        await monitor.run_once()
        recovery = await monitor.run_once()
        no_event = await monitor.run_once()

        assert len(recovery) == 1
        assert isinstance(recovery[0], ModelProjectionRecoveredEvent)
        assert len(no_event) == 0


# ---------------------------------------------------------------------------
# Skip contracts without freshness fields
# ---------------------------------------------------------------------------


class TestSkipWithoutFreshnessFields:
    @pytest.mark.asyncio
    async def test_skips_contract_without_freshness_field(self) -> None:
        # ModelProjectionContract requires min_length=1 for freshness_field,
        # so we mock the field being empty via a contract with a real field
        # but patch the attribute at runtime to simulate the absent-field path.
        contract = _make_contract()
        call_count = 0

        async def query(table: str, field: str) -> datetime:
            nonlocal call_count
            call_count += 1
            return datetime.now(UTC)

        monitor = _make_monitor((contract,), query)

        # Temporarily override freshness_field to empty string via object.__setattr__
        # since the model is frozen.
        object.__setattr__(contract, "freshness_field", "")
        events = await monitor.run_once()

        assert events == []
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_skips_contract_without_source_table(self) -> None:
        contract = _make_contract()
        call_count = 0

        async def query(table: str, field: str) -> datetime:
            nonlocal call_count
            call_count += 1
            return datetime.now(UTC)

        monitor = _make_monitor((contract,), query)

        object.__setattr__(contract, "freshness_source_table", "")
        events = await monitor.run_once()

        assert events == []
        assert call_count == 0


# ---------------------------------------------------------------------------
# NULL / error query paths
# ---------------------------------------------------------------------------


class TestQueryEdgeCases:
    @pytest.mark.asyncio
    async def test_null_query_result_skipped(self) -> None:
        contract = _make_contract(sla_seconds=30)

        async def query(table: str, field: str) -> None:
            return None

        monitor = _make_monitor((contract,), query)
        events = await monitor.run_once()

        assert events == []

    @pytest.mark.asyncio
    async def test_query_failure_swallowed_loop_continues(self) -> None:
        contract_bad = _make_contract(
            name="bad", sla_seconds=30, freshness_source_table="bad_table"
        )
        now = datetime.now(UTC)
        good_ts = now - timedelta(seconds=60)
        contract_good = _make_contract(
            name="good", sla_seconds=30, freshness_source_table="good_table"
        )

        async def query(table: str, field: str) -> datetime:
            if table == "bad_table":
                raise RuntimeError("DB exploded")
            return good_ts

        monitor = _make_monitor((contract_bad, contract_good), query)
        events = await monitor.run_once()

        # bad projection query failed → skipped; good projection is stale → degraded
        assert len(events) == 1
        assert events[0].projection_name == "good"

    @pytest.mark.asyncio
    async def test_naive_timestamp_handled_without_error(self) -> None:
        now = datetime.now(UTC)
        naive_ts = (now - timedelta(seconds=60)).replace(tzinfo=None)
        contract = _make_contract(sla_seconds=30)

        async def query(table: str, field: str) -> datetime:
            return naive_ts

        monitor = _make_monitor((contract,), query)
        events = await monitor.run_once()

        assert len(events) == 1
        assert isinstance(events[0], ModelProjectionDegradedEvent)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ServiceFreshnessMonitor(
                contracts=(),
                query_fn=AsyncMock(),
                check_interval_seconds=0,
                topic_registry=_make_stub_registry(),
            )

    @pytest.mark.asyncio
    async def test_start_stop_idempotent(self) -> None:
        monitor = _make_monitor((), AsyncMock())
        await monitor.start()
        await monitor.start()
        await monitor.stop()
        await monitor.stop()
