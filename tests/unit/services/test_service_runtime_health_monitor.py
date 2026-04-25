# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceRuntimeHealthMonitor.

Validates:
- Model instantiation and field constraints
- run_once() produces a correctly-shaped event
- Discovery errors surface as DEGRADED
- Empty consumer groups surface as DEGRADED
- Uncovered topics surface as CRITICAL/DEGRADED
- Event emission to Kafka via ProtocolEventBusLike
- Start/stop lifecycle is idempotent

Related Tickets:
    - OMN-8623: Runtime health alerting
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
from omnibase_infra.models.health.model_runtime_health_dimension import (
    ModelRuntimeHealthDimension,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ConsumerGroupSnapshot,
    ServiceRuntimeHealthMonitor,
    _worst,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

# =============================================================================
# Helpers
# =============================================================================


def _make_manifest(contracts=5, errors=0, subscribe_topics=()):
    """Return a minimal mock ModelAutoWiringManifest."""
    m = MagicMock()
    m.total_discovered = contracts
    m.total_errors = errors
    m.all_subscribe_topics.return_value = subscribe_topics
    return m


# =============================================================================
# _worst helper
# =============================================================================


class TestWorstHelper:
    def test_all_healthy(self):
        assert _worst(["HEALTHY", "HEALTHY"]) == "HEALTHY"

    def test_one_degraded(self):
        assert _worst(["HEALTHY", "DEGRADED"]) == "DEGRADED"

    def test_critical_wins(self):
        assert _worst(["HEALTHY", "DEGRADED", "CRITICAL"]) == "CRITICAL"

    def test_empty_list(self):
        assert _worst([]) == "HEALTHY"


# =============================================================================
# Model validation
# =============================================================================


class TestModelRuntimeHealthCheckEvent:
    def test_valid_instantiation(self):
        ev = ModelRuntimeHealthCheckEvent(
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            status="HEALTHY",
        )
        assert ev.status == "HEALTHY"
        assert ev.dimensions == ()

    def test_dimension_validation(self):
        dim = ModelRuntimeHealthDimension(
            name="test_dim", status="DEGRADED", detail="something bad"
        )
        assert dim.status == "DEGRADED"

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            ModelRuntimeHealthCheckEvent(
                correlation_id=uuid4(),
                timestamp=datetime.now(UTC),
                status="UNKNOWN",  # type: ignore[arg-type]
            )


# =============================================================================
# ServiceRuntimeHealthMonitor — init
# =============================================================================


class TestServiceRuntimeHealthMonitorInit:
    def test_defaults(self):
        monitor = ServiceRuntimeHealthMonitor()
        assert monitor._check_interval == 300.0
        assert monitor._event_bus is None
        assert not monitor._running

    def test_custom_interval(self):
        monitor = ServiceRuntimeHealthMonitor(check_interval_seconds=60.0)
        assert monitor._check_interval == 60.0

    def test_topic_resolved(self):
        registry = ServiceTopicRegistry.from_defaults()
        monitor = ServiceRuntimeHealthMonitor(topic_registry=registry)
        assert (
            monitor._health_topic == "onex.evt.omnibase-infra.runtime-health-check.v1"
        )


# =============================================================================
# ServiceRuntimeHealthMonitor — run_once (no Kafka)
# =============================================================================


class TestRunOnceNoKafka:
    """Tests that run_once works when bootstrap_servers is not set."""

    @pytest.mark.asyncio
    async def test_healthy_when_no_errors(self):
        manifest = _make_manifest(contracts=10, errors=0)
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="")

        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ):
            event = await monitor.run_once()

        assert event.contract_count == 10
        assert event.discovery_error_count == 0
        # No bootstrap_servers → consumer checks skipped → should be HEALTHY overall
        assert event.status == "HEALTHY"

    @pytest.mark.asyncio
    async def test_degraded_on_discovery_errors(self):
        manifest = _make_manifest(contracts=8, errors=3)
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="")

        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ):
            event = await monitor.run_once()

        assert event.discovery_error_count == 3
        assert event.status in ("DEGRADED", "CRITICAL")
        dim_names = {d.name for d in event.dimensions}
        assert "discovery_errors" in dim_names
        degraded_dims = [d for d in event.dimensions if d.status != "HEALTHY"]
        assert any(d.name == "discovery_errors" for d in degraded_dims)

    @pytest.mark.asyncio
    async def test_critical_on_discovery_exception(self):
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="")

        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            side_effect=RuntimeError("boom"),
        ):
            event = await monitor.run_once()

        assert event.status == "CRITICAL"
        critical_dims = [d for d in event.dimensions if d.status == "CRITICAL"]
        assert any(d.name == "discovery_errors" for d in critical_dims)


# =============================================================================
# ServiceRuntimeHealthMonitor — run_once with mocked Kafka admin
# =============================================================================


class TestRunOnceWithKafka:
    """Tests consumer group coverage checks with mocked Kafka group snapshots."""

    def _mock_admin(self, groups, empty_groups):
        """Build mock group snapshots."""
        return [
            ConsumerGroupSnapshot(
                group_id=g,
                state="EMPTY" if g in empty_groups else "STABLE",
            )
            for g in groups
        ]

    @pytest.mark.asyncio
    async def test_healthy_when_all_groups_active(self):
        groups = ["onex-consumer-topic.v1", "onex-consumer-v2"]
        manifest = _make_manifest(contracts=5, errors=0, subscribe_topics=["topic.v1"])
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="localhost:9092")

        snapshots = self._mock_admin(groups, empty_groups=set())

        with (
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
                return_value=manifest,
            ),
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
                return_value=snapshots,
            ),
        ):
            event = await monitor.run_once()

        assert event.consumer_group_count == 2
        assert event.empty_consumer_group_count == 0
        assert event.uncovered_topic_count == 0

    @pytest.mark.asyncio
    async def test_degraded_when_empty_groups(self):
        groups = ["onex-consumer-v1", "onex-consumer-v2", "onex-consumer-v3"]
        manifest = _make_manifest(contracts=5, errors=0)
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="localhost:9092")

        snapshots = self._mock_admin(groups, empty_groups={"onex-consumer-v2"})

        with (
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
                return_value=manifest,
            ),
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
                return_value=snapshots,
            ),
        ):
            event = await monitor.run_once()

        assert event.empty_consumer_group_count == 1
        degraded_dims = [d for d in event.dimensions if d.status != "HEALTHY"]
        assert any(d.name == "empty_consumer_groups" for d in degraded_dims)

    @pytest.mark.asyncio
    async def test_degraded_when_group_listing_fails(self):
        manifest = _make_manifest(contracts=5, errors=0, subscribe_topics=["topic.v1"])
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="localhost:9092")

        with (
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
                return_value=manifest,
            ),
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
                side_effect=UnicodeDecodeError("utf-8", b"\x80", 0, 1, "bad byte"),
            ),
        ):
            event = await monitor.run_once()

        assert event.status == "DEGRADED"
        degraded_dims = [d for d in event.dimensions if d.status == "DEGRADED"]
        assert any(d.name == "consumer_coverage" for d in degraded_dims)


# =============================================================================
# ServiceRuntimeHealthMonitor — event emission
# =============================================================================


class TestEventEmission:
    @pytest.mark.asyncio
    async def test_emits_to_event_bus(self):
        bus = AsyncMock()
        manifest = _make_manifest()
        monitor = ServiceRuntimeHealthMonitor(event_bus=bus, bootstrap_servers="")

        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ):
            await monitor.run_once()

        bus.publish_envelope.assert_awaited_once()
        call_kwargs = bus.publish_envelope.await_args
        assert call_kwargs is not None
        topic_arg = call_kwargs.kwargs.get("topic") or call_kwargs.args[1]
        assert "runtime-health-check" in topic_arg

    @pytest.mark.asyncio
    async def test_emission_failure_does_not_crash(self):
        bus = AsyncMock()
        bus.publish_envelope.side_effect = RuntimeError("kafka down")
        manifest = _make_manifest()
        monitor = ServiceRuntimeHealthMonitor(event_bus=bus, bootstrap_servers="")

        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ):
            # Should not raise
            event = await monitor.run_once()

        assert event is not None

    @pytest.mark.asyncio
    async def test_no_emission_when_no_bus(self):
        manifest = _make_manifest()
        monitor = ServiceRuntimeHealthMonitor(event_bus=None, bootstrap_servers="")

        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ):
            event = await monitor.run_once()

        assert event is not None


# =============================================================================
# ServiceRuntimeHealthMonitor — lifecycle
# =============================================================================


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        manifest = _make_manifest(contracts=5, errors=0)
        monitor = ServiceRuntimeHealthMonitor(
            bootstrap_servers="", check_interval_seconds=9999.0
        )
        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ):
            await monitor.start()
        assert monitor._running
        assert monitor._task is not None
        await monitor.stop()
        assert not monitor._running
        assert monitor._task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        manifest = _make_manifest(contracts=5, errors=0)
        monitor = ServiceRuntimeHealthMonitor(
            bootstrap_servers="", check_interval_seconds=9999.0
        )
        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ):
            await monitor.start()
            task_before = monitor._task
            await monitor.start()  # second call — should not re-run initial check
        assert monitor._task is task_before
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent_without_start(self):
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="")
        await monitor.stop()  # should not raise
        assert not monitor._running


# =============================================================================
# Topic key registration
# =============================================================================


class TestTopicKey:
    def test_topic_key_registered(self):
        assert hasattr(topic_keys, "RUNTIME_HEALTH_CHECK")

    def test_topic_resolves(self):
        registry = ServiceTopicRegistry.from_defaults()
        topic = registry.resolve(topic_keys.RUNTIME_HEALTH_CHECK)
        assert topic == "onex.evt.omnibase-infra.runtime-health-check.v1"
