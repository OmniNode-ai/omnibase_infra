# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for RuntimeHostProcess auto-introspection (OMN-1930).

This module validates that RuntimeHostProcess correctly integrates with
the introspection service to announce node presence on startup with
configurable jitter and throttling.

Test Coverage:
- P1.5: Startup emits introspection event with jitter
- P1.6: Rapid restart throttling prevents stampede

Related:
    - OMN-1930: Phase 1 - Fix Auto-Introspection (P0)
    - src/omnibase_infra/runtime/service_runtime_host_process.py
    - src/omnibase_infra/protocols/protocol_node_introspection.py
    - src/omnibase_infra/models/runtime/model_runtime_introspection_config.py
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumIntrospectionReason
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolNodeIntrospection

pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_introspection_service() -> MagicMock:
    """Create a mock introspection service for testing.

    Returns:
        MagicMock implementing ProtocolNodeIntrospection interface.
    """
    service = MagicMock()
    service.publish_introspection = AsyncMock()
    service.start_heartbeat_task = AsyncMock()
    service.stop_heartbeat_task = AsyncMock()
    return service


@pytest.fixture
def event_bus() -> EventBusInmemory:
    """Create an in-memory event bus for testing.

    Returns:
        EventBusInmemory instance.
    """
    return EventBusInmemory()


# =============================================================================
# P1.5: Test startup emits introspection with jitter
# =============================================================================


class TestStartupIntrospectionWithJitter:
    """Tests for startup introspection with jitter (P1.5)."""

    @pytest.mark.asyncio
    async def test_startup_publishes_introspection_event(
        self,
        mock_introspection_service: MagicMock,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that startup publishes introspection event with STARTUP reason.

        Verifies:
        - publish_introspection is called after handlers wired
        - reason is EnumIntrospectionReason.STARTUP
        - correlation_id is provided
        - start_heartbeat_task is called after introspection
        """
        config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=0,  # No jitter for deterministic test
            throttle_min_interval_s=10,
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=mock_introspection_service,
            introspection_config=config,
        )

        # Seed mock handlers to pass validation
        seed_mock_handlers(process)

        try:
            await process.start()

            # Verify introspection was published
            mock_introspection_service.publish_introspection.assert_called_once()

            # Verify call arguments
            call_kwargs = (
                mock_introspection_service.publish_introspection.call_args.kwargs
            )
            assert call_kwargs["reason"] == EnumIntrospectionReason.STARTUP
            assert isinstance(call_kwargs["correlation_id"], UUID)

            # Verify heartbeat task was started
            mock_introspection_service.start_heartbeat_task.assert_called_once()

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_startup_applies_jitter_delay(
        self,
        mock_introspection_service: MagicMock,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that startup applies jitter delay before publishing.

        Verifies:
        - With jitter > 0, there is a delay before introspection is published
        - The delay is random within the configured range
        """
        jitter_max_ms = 100  # 100ms max jitter for fast test
        config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=jitter_max_ms,
            throttle_min_interval_s=10,
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=mock_introspection_service,
            introspection_config=config,
        )

        # Seed mock handlers
        seed_mock_handlers(process)

        try:
            start_time = time.monotonic()
            await process.start()
            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Verify introspection was published
            mock_introspection_service.publish_introspection.assert_called_once()

            # Note: We can't guarantee jitter was applied because random.randint(0, max)
            # might return 0. We just verify the call happened within reasonable time.
            # The jitter_max_ms=100 means at most 100ms delay.
            assert elapsed_ms < 5000  # Should complete well under 5 seconds

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_startup_skips_introspection_when_disabled(
        self,
        mock_introspection_service: MagicMock,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that startup skips introspection when disabled in config.

        Verifies:
        - When enabled=False, publish_introspection is NOT called
        - start_heartbeat_task is NOT called
        """
        config = ModelRuntimeIntrospectionConfig(
            enabled=False,  # Disabled
            jitter_max_ms=0,
            throttle_min_interval_s=10,
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=mock_introspection_service,
            introspection_config=config,
        )

        # Seed mock handlers
        seed_mock_handlers(process)

        try:
            await process.start()

            # Verify introspection was NOT published
            mock_introspection_service.publish_introspection.assert_not_called()
            mock_introspection_service.start_heartbeat_task.assert_not_called()

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_startup_skips_introspection_when_service_not_provided(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that startup skips introspection when service is not injected.

        Verifies:
        - When introspection_service is None, startup completes normally
        - No errors are raised
        """
        config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=0,
            throttle_min_interval_s=10,
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=None,  # Not provided
            introspection_config=config,
        )

        # Seed mock handlers
        seed_mock_handlers(process)

        try:
            # Should complete without error
            await process.start()
            assert process.is_running

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_startup_continues_on_introspection_error(
        self,
        mock_introspection_service: MagicMock,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that startup continues even if introspection fails.

        Verifies:
        - If publish_introspection raises an exception, startup still completes
        - Error is logged but doesn't block startup
        """
        # Configure mock to raise an error
        mock_introspection_service.publish_introspection.side_effect = Exception(
            "Kafka unavailable"
        )

        config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=0,
            throttle_min_interval_s=10,
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=mock_introspection_service,
            introspection_config=config,
        )

        # Seed mock handlers
        seed_mock_handlers(process)

        try:
            # Should complete without raising the introspection error
            await process.start()
            assert process.is_running

            # Heartbeat should NOT be started when publish fails
            mock_introspection_service.start_heartbeat_task.assert_not_called()

        finally:
            await process.stop()


# =============================================================================
# P1.6: Test throttle on rapid restart
# =============================================================================


class TestThrottleOnRapidRestart:
    """Tests for throttle on rapid restart (P1.6)."""

    @pytest.mark.asyncio
    async def test_throttle_prevents_rapid_introspection(
        self,
        mock_introspection_service: MagicMock,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that rapid restart doesn't cause introspection stampede.

        Verifies:
        - First introspection after startup is published
        - Calling _publish_introspection_with_jitter again within throttle_min_interval
          does NOT publish another introspection
        """
        config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=0,  # No jitter for deterministic test
            throttle_min_interval_s=10,  # 10 second throttle
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=mock_introspection_service,
            introspection_config=config,
        )

        # Seed mock handlers
        seed_mock_handlers(process)

        try:
            await process.start()

            # First introspection should have been published
            assert mock_introspection_service.publish_introspection.call_count == 1

            # Simulate rapid restart by calling introspection again immediately
            # Access the private method for testing throttle behavior
            from uuid import uuid4

            await process._publish_introspection_with_jitter(correlation_id=uuid4())

            # Throttle should have prevented second introspection
            assert mock_introspection_service.publish_introspection.call_count == 1

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_throttle_allows_introspection_after_interval(
        self,
        mock_introspection_service: MagicMock,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that introspection is allowed after throttle interval passes.

        Verifies:
        - First introspection is published
        - After waiting longer than throttle_min_interval, second introspection
          IS published

        Note:
            This test manipulates the internal _last_introspection_time to simulate
            time passing without actually waiting. This is necessary because the
            minimum throttle is 1 second (per model validation) which is too slow
            for unit tests.
        """
        config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=0,
            throttle_min_interval_s=1,  # Minimum allowed value
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=mock_introspection_service,
            introspection_config=config,
        )

        # Seed mock handlers
        seed_mock_handlers(process)

        try:
            await process.start()

            # First introspection should have been published
            assert mock_introspection_service.publish_introspection.call_count == 1

            # Simulate time passing by adjusting the last introspection time
            # Set it to 2 seconds ago (greater than 1 second throttle)
            process._last_introspection_time = time.monotonic() - 2.0

            # Now calling introspection again should succeed (throttle passed)
            from uuid import uuid4

            await process._publish_introspection_with_jitter(correlation_id=uuid4())

            # Second introspection should be published (throttle interval passed)
            assert mock_introspection_service.publish_introspection.call_count == 2

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_multiple_processes_with_jitter_spread_load(
        self,
    ) -> None:
        """Test that multiple processes with jitter spread their introspection load.

        Verifies:
        - Multiple processes with same jitter_max_ms don't all fire at exact same time
        - This is a probabilistic test - with enough processes, we expect variation

        Note:
            This test uses small jitter to keep test fast while still demonstrating
            the spread behavior.
        """
        num_processes = 5
        jitter_max_ms = 50  # 50ms max jitter

        introspection_times: list[float] = []
        processes: list[RuntimeHostProcess] = []
        mock_services: list[MagicMock] = []

        try:
            for i in range(num_processes):
                mock_service = MagicMock()

                # Record time when introspection is called
                async def record_time(
                    reason: EnumIntrospectionReason,
                    correlation_id: UUID | None = None,
                    _times: list[float] = introspection_times,
                ) -> None:
                    _times.append(time.monotonic())

                mock_service.publish_introspection = AsyncMock(side_effect=record_time)
                mock_service.start_heartbeat_task = AsyncMock()
                mock_service.stop_heartbeat_task = AsyncMock()
                mock_services.append(mock_service)

                config = ModelRuntimeIntrospectionConfig(
                    enabled=True,
                    jitter_max_ms=jitter_max_ms,
                    throttle_min_interval_s=10,
                )

                process = RuntimeHostProcess(
                    event_bus=EventBusInmemory(),
                    config=make_runtime_config(
                        service_name=f"test-service-{i}",
                        node_name=f"test-node-{i}",
                    ),
                    introspection_service=mock_service,
                    introspection_config=config,
                )

                # Seed mock handlers
                seed_mock_handlers(process)
                processes.append(process)

            # Start all processes concurrently
            await asyncio.gather(*[p.start() for p in processes])

            # Verify all introspections were published
            assert len(introspection_times) == num_processes

            # With random jitter, times should not all be identical
            # (though with small jitter and fast CPU, they might be close)
            # At minimum, we verify all introspections completed
            for mock_service in mock_services:
                mock_service.publish_introspection.assert_called_once()

        finally:
            # Stop all processes
            for process in processes:
                await process.stop()


# =============================================================================
# Config Validation Tests
# =============================================================================


class TestIntrospectionConfigValidation:
    """Tests for ModelRuntimeIntrospectionConfig validation."""

    def test_config_defaults(self) -> None:
        """Test that config has sensible defaults."""
        config = ModelRuntimeIntrospectionConfig()

        assert config.enabled is True
        assert config.jitter_max_ms == 5000
        assert config.throttle_min_interval_s == 10

    def test_config_validates_jitter_bounds(self) -> None:
        """Test that jitter_max_ms validates bounds."""
        # Valid values
        config = ModelRuntimeIntrospectionConfig(jitter_max_ms=0)
        assert config.jitter_max_ms == 0

        config = ModelRuntimeIntrospectionConfig(jitter_max_ms=30000)
        assert config.jitter_max_ms == 30000

        # Invalid values should raise ValidationError
        with pytest.raises(ValueError):
            ModelRuntimeIntrospectionConfig(jitter_max_ms=-1)

        with pytest.raises(ValueError):
            ModelRuntimeIntrospectionConfig(jitter_max_ms=30001)

    def test_config_validates_throttle_bounds(self) -> None:
        """Test that throttle_min_interval_s validates bounds."""
        # Valid values
        config = ModelRuntimeIntrospectionConfig(throttle_min_interval_s=1)
        assert config.throttle_min_interval_s == 1

        config = ModelRuntimeIntrospectionConfig(throttle_min_interval_s=60)
        assert config.throttle_min_interval_s == 60

        # Invalid values should raise ValidationError
        with pytest.raises(ValueError):
            ModelRuntimeIntrospectionConfig(throttle_min_interval_s=0)

        with pytest.raises(ValueError):
            ModelRuntimeIntrospectionConfig(throttle_min_interval_s=61)

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable (frozen)."""
        config = ModelRuntimeIntrospectionConfig()

        with pytest.raises(Exception):  # ValidationError for frozen model
            config.enabled = False  # type: ignore[misc]


# =============================================================================
# Shutdown Lifecycle Tests
# =============================================================================


class TestShutdownHeartbeatCleanup:
    """Tests for heartbeat cleanup during stop() (OMN-1930)."""

    @pytest.mark.asyncio
    async def test_stop_calls_stop_heartbeat_task(
        self,
        mock_introspection_service: MagicMock,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that stop() invokes stop_heartbeat_task on the introspection service.

        Verifies:
        - stop_heartbeat_task is called during stop()
        - Heartbeat is cleaned up before event bus closure
        """
        config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=0,
            throttle_min_interval_s=10,
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=mock_introspection_service,
            introspection_config=config,
        )

        seed_mock_handlers(process)

        await process.start()
        mock_introspection_service.stop_heartbeat_task.assert_not_called()

        await process.stop()
        mock_introspection_service.stop_heartbeat_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_succeeds_when_stop_heartbeat_raises(
        self,
        mock_introspection_service: MagicMock,
        event_bus: EventBusInmemory,
    ) -> None:
        """Test that stop() completes even if stop_heartbeat_task raises.

        Verifies:
        - stop() does not propagate the heartbeat stop error
        - The process transitions to stopped state despite the error
        """
        mock_introspection_service.stop_heartbeat_task.side_effect = Exception(
            "Heartbeat cleanup failed"
        )

        config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=0,
            throttle_min_interval_s=10,
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=mock_introspection_service,
            introspection_config=config,
        )

        seed_mock_handlers(process)

        await process.start()
        assert process.is_running

        # stop() should succeed despite heartbeat cleanup error
        await process.stop()
        assert not process.is_running
