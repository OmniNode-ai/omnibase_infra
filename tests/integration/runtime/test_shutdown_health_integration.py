# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for shutdown and health check interaction.

Tests the interaction between RuntimeHostProcess shutdown lifecycle and
health check behavior. These tests verify:

1. Health check behavior during shutdown
2. Health check behavior after shutdown
3. Health server HTTP responses during runtime shutdown
4. Graceful shutdown sequence with no errors

These tests use real instances (not mocks) to verify integration behavior.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import aiohttp
import pytest

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.runtime.health_server import HealthServer
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess


class TestShutdownHealthIntegration:
    """Integration tests for shutdown and health check interaction."""

    @pytest.mark.asyncio
    async def test_health_check_after_shutdown_shows_not_running(self) -> None:
        """Health check after shutdown should show is_running=False.

        This test verifies that after RuntimeHostProcess.stop() completes,
        the health_check() method correctly reports:
        - is_running: False
        - healthy: False (since not running)
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        # failures from the singleton registry affecting the test
        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()

            # Verify running state first
            health_before = await runtime.health_check()
            assert health_before["is_running"] is True
            assert health_before["healthy"] is True

            # Act - stop the runtime
            await runtime.stop()
            health_after = await runtime.health_check()

            # Assert
            assert health_after["is_running"] is False
            assert health_after["healthy"] is False
            # Degraded should be False (requires running state)
            assert health_after["degraded"] is False

    @pytest.mark.asyncio
    async def test_health_check_during_shutdown_transition(self) -> None:
        """Health check during shutdown should handle transitional state.

        This test verifies that calling health_check() while shutdown is
        in progress does not raise exceptions and returns a consistent state.

        Note: Due to the fast nature of InMemoryEventBus shutdown, we may
        not catch the exact transitional state, but we verify no exceptions
        occur and the final state is correct.
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()

            # Act - start shutdown (but don't await immediately)
            # Create a task for shutdown
            shutdown_task = asyncio.create_task(runtime.stop())

            # Call health_check during/immediately after shutdown starts
            # This should not raise any exceptions
            try:
                health = await runtime.health_check()
                # Health check should return a valid dict
                assert isinstance(health, dict)
                assert "is_running" in health
                assert "healthy" in health
            except Exception as e:
                pytest.fail(f"Health check during shutdown raised exception: {e}")

            # Wait for shutdown to complete
            await shutdown_task

            # Verify final state
            final_health = await runtime.health_check()
            assert final_health["is_running"] is False
            assert final_health["healthy"] is False

    @pytest.mark.asyncio
    async def test_health_server_returns_503_when_runtime_stopped(self) -> None:
        """Health server should return 503 when runtime is stopped.

        This test verifies the HTTP health endpoint behavior when:
        1. Runtime is started and health server is running
        2. Runtime is stopped (but health server remains running)
        3. HTTP request to /health should return 503 (unhealthy)
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)
        # Use port 0 for automatic port assignment to avoid conflicts
        health_server = HealthServer(runtime=runtime, port=0, version="test-1.0.0")

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()
            await health_server.start()

            # Get actual port after binding
            site = health_server._site
            assert site is not None
            internal_server = site._server
            assert internal_server is not None
            sockets = getattr(internal_server, "sockets", None)
            assert sockets is not None and len(sockets) > 0
            actual_port: int = sockets[0].getsockname()[1]

            try:
                # Verify health returns 200 when runtime is running
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{actual_port}/health"
                    ) as response:
                        assert response.status == 200
                        data = await response.json()
                        assert data["status"] == "healthy"

                # Act - stop runtime but keep health server running
                await runtime.stop()

                # Health server should now return 503 (unhealthy)
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{actual_port}/health"
                    ) as response:
                        status = response.status
                        body = await response.json()

                # Assert
                assert status == 503
                assert body["status"] == "unhealthy"
                assert body["details"]["is_running"] is False
                assert body["details"]["healthy"] is False

            finally:
                # Cleanup
                await health_server.stop()

    @pytest.mark.asyncio
    async def test_health_server_returns_503_after_runtime_shutdown(self) -> None:
        """Health server should return 503 after runtime shutdown completes.

        Similar to test_health_server_returns_503_when_runtime_stopped but
        explicitly tests the complete shutdown flow.
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)
        health_server = HealthServer(runtime=runtime, port=0, version="test-1.0.0")

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()
            await health_server.start()

            # Get actual port
            site = health_server._site
            assert site is not None
            internal_server = site._server
            assert internal_server is not None
            sockets = getattr(internal_server, "sockets", None)
            assert sockets is not None and len(sockets) > 0
            actual_port: int = sockets[0].getsockname()[1]

            try:
                # Stop runtime completely
                await runtime.stop()

                # Multiple health checks after shutdown should all return 503
                async with aiohttp.ClientSession() as session:
                    for _ in range(3):
                        async with session.get(
                            f"http://127.0.0.1:{actual_port}/health"
                        ) as response:
                            assert response.status == 503
                            data = await response.json()
                            assert data["status"] == "unhealthy"
                            assert data["details"]["is_running"] is False

            finally:
                await health_server.stop()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_sequence_no_errors(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Graceful shutdown sequence should produce no errors.

        This test verifies the complete shutdown sequence:
        1. Start runtime + health server
        2. Verify health returns 200
        3. Stop runtime first (order matters for clean shutdown)
        4. Stop health server
        5. Verify no ERROR level logs occurred
        """
        import logging

        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)
        health_server = HealthServer(runtime=runtime, port=0, version="test-1.0.0")

        async def noop_populate() -> None:
            pass

        with caplog.at_level(logging.ERROR):
            with patch.object(
                runtime, "_populate_handlers_from_registry", noop_populate
            ):
                # Start both
                await runtime.start()
                await health_server.start()

                # Get actual port
                site = health_server._site
                assert site is not None
                internal_server = site._server
                assert internal_server is not None
                sockets = getattr(internal_server, "sockets", None)
                assert sockets is not None and len(sockets) > 0
                actual_port: int = sockets[0].getsockname()[1]

                # Verify health returns 200
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{actual_port}/health"
                    ) as response:
                        assert response.status == 200

                # Graceful shutdown - stop runtime first, then health server
                await runtime.stop()
                await health_server.stop()

        # Assert no ERROR logs
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert (
            len(error_logs) == 0
        ), f"Unexpected errors during shutdown: {[r.message for r in error_logs]}"

        # Verify both are stopped
        assert runtime.is_running is False
        assert health_server.is_running is False

    @pytest.mark.asyncio
    async def test_shutdown_then_health_check_multiple_times(self) -> None:
        """Health check can be called multiple times after shutdown.

        Verifies that health_check() is idempotent and can be called
        repeatedly after shutdown without causing issues.
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()
            await runtime.stop()

            # Act - call health_check multiple times
            results = []
            for _ in range(5):
                health = await runtime.health_check()
                results.append(health)

            # Assert - all results should be consistent
            for health in results:
                assert health["is_running"] is False
                assert health["healthy"] is False
                assert health["degraded"] is False

    @pytest.mark.asyncio
    async def test_event_bus_health_after_runtime_shutdown(self) -> None:
        """Event bus health should reflect closed state after shutdown.

        When RuntimeHostProcess.stop() is called, the event bus should
        also be closed, and its health_check() should reflect this.
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()

            # Verify event bus is started
            eb_health_before = await event_bus.health_check()
            assert eb_health_before["started"] is True
            assert eb_health_before["healthy"] is True

            # Stop runtime
            await runtime.stop()

            # Event bus should now be closed
            eb_health_after = await event_bus.health_check()
            assert eb_health_after["started"] is False
            assert eb_health_after["healthy"] is False

    @pytest.mark.asyncio
    async def test_health_check_reflects_degraded_false_after_shutdown(self) -> None:
        """Degraded status should be False after shutdown (not running).

        The degraded state only applies when the process is running but
        with reduced functionality. After shutdown, degraded should be False.
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()

            # Simulate a degraded state while running
            runtime._failed_handlers = {"test_handler": "Test failure"}
            health_running = await runtime.health_check()

            # Should be degraded while running
            assert health_running["is_running"] is True
            assert health_running["degraded"] is True
            assert health_running["healthy"] is False

            # Stop runtime
            await runtime.stop()

            # After shutdown, degraded should be False
            health_stopped = await runtime.health_check()
            assert health_stopped["is_running"] is False
            assert health_stopped["degraded"] is False
            assert health_stopped["healthy"] is False


class TestHealthServerShutdownBehavior:
    """Tests for HealthServer behavior during various shutdown scenarios."""

    @pytest.mark.asyncio
    async def test_health_server_stop_idempotent(self) -> None:
        """Health server stop() should be idempotent.

        Calling stop() multiple times should not raise exceptions.
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)
        health_server = HealthServer(runtime=runtime, port=0)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()
            await health_server.start()

            # Act - stop multiple times
            await health_server.stop()
            await health_server.stop()
            await health_server.stop()

            # Assert - no exceptions, server is stopped
            assert health_server.is_running is False

            # Cleanup
            await runtime.stop()

    @pytest.mark.asyncio
    async def test_health_server_handles_runtime_not_started(self) -> None:
        """Health server should handle runtime that was never started.

        When runtime was never started, health endpoint should return 503.
        """
        # Arrange - runtime never started
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)
        health_server = HealthServer(runtime=runtime, port=0)

        await health_server.start()

        # Get actual port
        site = health_server._site
        assert site is not None
        internal_server = site._server
        assert internal_server is not None
        sockets = getattr(internal_server, "sockets", None)
        assert sockets is not None and len(sockets) > 0
        actual_port: int = sockets[0].getsockname()[1]

        try:
            # Act - request health when runtime never started
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{actual_port}/health"
                ) as response:
                    status = response.status
                    data = await response.json()

            # Assert
            assert status == 503
            assert data["status"] == "unhealthy"
            assert data["details"]["is_running"] is False

        finally:
            await health_server.stop()

    @pytest.mark.asyncio
    async def test_ready_endpoint_same_as_health_after_shutdown(self) -> None:
        """The /ready endpoint should return same status as /health after shutdown.

        Both endpoints are aliases and should return 503 after runtime shutdown.
        """
        # Arrange
        event_bus = InMemoryEventBus()
        runtime = RuntimeHostProcess(event_bus=event_bus)
        health_server = HealthServer(runtime=runtime, port=0)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            await runtime.start()
            await health_server.start()

            # Get actual port
            site = health_server._site
            assert site is not None
            internal_server = site._server
            assert internal_server is not None
            sockets = getattr(internal_server, "sockets", None)
            assert sockets is not None and len(sockets) > 0
            actual_port: int = sockets[0].getsockname()[1]

            try:
                # Stop runtime
                await runtime.stop()

                # Both endpoints should return 503
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{actual_port}/health"
                    ) as health_response:
                        health_status = health_response.status
                        health_data = await health_response.json()

                    async with session.get(
                        f"http://127.0.0.1:{actual_port}/ready"
                    ) as ready_response:
                        ready_status = ready_response.status
                        ready_data = await ready_response.json()

                # Assert both return same status
                assert health_status == 503
                assert ready_status == 503
                assert health_data["status"] == "unhealthy"
                assert ready_data["status"] == "unhealthy"

            finally:
                await health_server.stop()


__all__: list[str] = [
    "TestShutdownHealthIntegration",
    "TestHealthServerShutdownBehavior",
]
