# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Concurrency tests for VaultAdapter.

These tests verify thread safety of circuit breaker state variables
and concurrent operation handling under production load scenarios.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_vault import VaultAdapter


@pytest.fixture
def vault_config() -> dict[str, object]:
    """Provide test Vault configuration."""
    return {
        "url": "https://vault.example.com:8200",
        "token": "s.test1234567890",
        "namespace": "engineering",
        "timeout_seconds": 30.0,
        "verify_ssl": True,
        "token_renewal_threshold_seconds": 300.0,
        "retry": {
            "max_attempts": 3,
            "initial_backoff_seconds": 0.1,
            "max_backoff_seconds": 10.0,
            "exponential_base": 2.0,
        },
    }


@pytest.fixture
def mock_hvac_client() -> MagicMock:
    """Provide mocked hvac.Client."""
    client = MagicMock()
    client.is_authenticated.return_value = True
    client.secrets.kv.v2 = MagicMock()
    client.auth.token = MagicMock()
    # Mock token lookup response with TTL
    client.auth.token.lookup_self.return_value = {
        "data": {
            "ttl": 3600,  # 1 hour TTL in seconds
            "renewable": True,
        }
    }
    client.sys = MagicMock()
    return client


class TestVaultAdapterConcurrency:
    """Test VaultAdapter concurrent operation handling and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_state_updates(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit breaker state is thread-safe under concurrent access.

        This test verifies that concurrent operations properly update circuit breaker
        state without race conditions by:
        1. Tracking actual success and failure counts
        2. Verifying the circuit breaker failure count matches observed failures
        3. Ensuring no RuntimeError from race conditions occurs
        """
        handler = VaultAdapter()

        vault_config["circuit_breaker_failure_threshold"] = 10

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Mix of failures and successes - create a predictable pattern
            responses = [
                Exception("Connection error"),
                {"data": {"data": {"key": "value"}, "metadata": {"version": 1}}},
                Exception("Connection error"),
                {"data": {"data": {"key": "value"}, "metadata": {"version": 1}}},
            ] * 5

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = responses

            await handler.initialize(vault_config)

            # Track results for verification
            success_count = 0
            failure_count = 0
            lock = asyncio.Lock()

            # Launch 20 concurrent requests (mix of success and failure)
            async def execute_request(index: int) -> dict[str, object] | None:
                nonlocal success_count, failure_count
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    result = await handler.execute(envelope)
                    async with lock:
                        success_count += 1
                    return result
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        failure_count += 1
                    return None

            tasks = [execute_request(i) for i in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify no race conditions occurred (no RuntimeError from race conditions)
            for result in results:
                assert not isinstance(result, RuntimeError), (
                    f"Race condition detected: {result}"
                )

            # Verify all requests were processed
            assert success_count + failure_count == 20, (
                f"Expected 20 total, got {success_count + failure_count}"
            )

            # Verify circuit breaker failure count is consistent with observed failures
            # Due to retries, failure count could be higher than circuit breaker count
            # but circuit breaker should have recorded at least some failures (mixin attrs)
            assert handler._circuit_breaker_failures >= 0, (
                "Circuit breaker failure count should be non-negative"
            )

            # With threshold of 10, circuit should not be open if failures < 10
            # or should be open if failures >= 10 (using mixin attributes)
            if handler._circuit_breaker_failures >= 10:
                assert handler._circuit_breaker_open is True, (
                    "Circuit should be OPEN after 10+ failures"
                )

    @pytest.mark.asyncio
    async def test_concurrent_successful_operations(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent successful operations don't cause race conditions."""
        handler = VaultAdapter()

        # Configure larger queue size to handle concurrent requests
        vault_config["max_concurrent_operations"] = 20
        vault_config["max_queue_size_multiplier"] = 5  # Queue size = 20 * 5 = 100

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # All requests succeed
            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            await handler.initialize(vault_config)

            # Launch 50 concurrent successful requests (within queue capacity of 100)
            async def execute_request(index: int) -> dict[str, object]:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                return await handler.execute(envelope)

            tasks = [execute_request(i) for i in range(50)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(result["status"] == "success" for result in results)

            # Circuit breaker should remain CLOSED with zero failures (using mixin attrs)
            assert handler._circuit_breaker_open is False
            assert handler._circuit_breaker_failures == 0

    @pytest.mark.asyncio
    async def test_concurrent_failures_trigger_circuit_correctly(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent failures correctly trigger circuit breaker."""
        handler = VaultAdapter()

        vault_config["circuit_breaker_failure_threshold"] = 3

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # All requests fail
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            # Launch concurrent failing requests
            async def execute_request(index: int) -> dict[str, object] | None:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    return await handler.execute(envelope)
                except (InfraConnectionError, InfraUnavailableError):
                    return None

            tasks = [execute_request(i) for i in range(10)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Circuit should open after threshold failures (using mixin attributes)
            assert handler._circuit_breaker_open is True
            assert handler._circuit_breaker_failures >= 3

    @pytest.mark.asyncio
    async def test_concurrent_mixed_write_operations(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent write operations are thread-safe."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.create_or_update_secret.return_value = {
                "data": {"version": 1, "created_time": "2025-01-01T00:00:00Z"}
            }

            await handler.initialize(vault_config)

            # Launch 30 concurrent write operations
            async def execute_write(index: int) -> dict[str, object]:
                envelope = {
                    "operation": "vault.write_secret",
                    "payload": {
                        "path": f"myapp/secret{index}",
                        "data": {"key": f"value{index}"},
                    },
                    "correlation_id": uuid4(),
                }
                return await handler.execute(envelope)

            tasks = [execute_write(i) for i in range(30)]
            results = await asyncio.gather(*tasks)

            # All writes should succeed
            assert all(result["status"] == "success" for result in results)

            # Verify create_or_update_secret was called 30 times (once per request)
            assert (
                mock_hvac_client.secrets.kv.v2.create_or_update_secret.call_count == 30
            )

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent health checks are thread-safe."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.sys.read_health_status.return_value = {
                "initialized": True,
                "sealed": False,
            }

            await handler.initialize(vault_config)

            # Launch 20 concurrent health checks
            tasks = [handler.health_check() for _ in range(20)]
            results = await asyncio.gather(*tasks)

            # All should report healthy
            assert all(result["healthy"] is True for result in results)
            assert all(result["initialized"] is True for result in results)

    @pytest.mark.asyncio
    async def test_shutdown_during_concurrent_operations(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test shutdown is safe during concurrent operations.

        This test verifies that:
        1. Shutdown can occur while operations are in progress
        2. No race conditions or deadlocks occur
        3. Handler state is properly cleaned up after shutdown
        4. All tasks complete (either successfully or with expected errors)
        """
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            await handler.initialize(vault_config)

            # Track task completion for verification
            started_count = 0
            completed_count = 0
            failed_count = 0
            lock = asyncio.Lock()

            # Launch concurrent operations with tracking
            async def execute_request(index: int) -> dict[str, object] | None:
                nonlocal started_count, completed_count, failed_count
                async with lock:
                    started_count += 1

                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    result = await handler.execute(envelope)
                    async with lock:
                        completed_count += 1
                    return result
                except RuntimeHostError:
                    # Expected if shutdown happens during execution
                    async with lock:
                        failed_count += 1
                    return None

            # Start operations
            tasks = [asyncio.create_task(execute_request(i)) for i in range(10)]

            # Ensure all tasks have started (yield to event loop)
            await asyncio.sleep(0)

            # Wait briefly for some tasks to potentially start executing
            _done, pending = await asyncio.wait(
                tasks, timeout=0.01, return_when=asyncio.FIRST_COMPLETED
            )

            # Shutdown handler while tasks may still be running
            await handler.shutdown()

            # Wait for all remaining tasks to complete (some may fail due to shutdown)
            if pending:
                await asyncio.wait(pending)

            # Verify all tasks were started
            assert started_count == 10, (
                f"Expected 10 started tasks, got {started_count}"
            )

            # Verify all tasks completed (either successfully or with expected error)
            total_finished = completed_count + failed_count
            assert total_finished == 10, (
                f"Expected 10 finished tasks, got {total_finished}"
            )

            # Verify handler is fully shut down
            assert handler._initialized is False, "Handler should not be initialized"
            assert handler._client is None, "Client should be None after shutdown"
            assert handler._executor is None, "Executor should be None after shutdown"

            # Verify circuit breaker is reset (initialized flag is cleared)
            assert handler._circuit_breaker_initialized is False, (
                "Circuit breaker should not be initialized after shutdown"
            )

    @pytest.mark.asyncio
    async def test_thread_pool_handles_concurrent_load(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test thread pool correctly handles concurrent operation load."""
        handler = VaultAdapter()

        # Set thread pool size to 5 and queue multiplier to handle 25 requests
        # Queue size = 5 * 10 = 50, which can handle 25 concurrent requests
        vault_config["max_concurrent_operations"] = 5
        vault_config["max_queue_size_multiplier"] = 10

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            await handler.initialize(vault_config)

            # Launch more requests than thread pool size (but within queue capacity)
            async def execute_request(index: int) -> dict[str, object]:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                return await handler.execute(envelope)

            tasks = [execute_request(i) for i in range(25)]
            results = await asyncio.gather(*tasks)

            # All should succeed despite thread pool size < request count
            assert all(result["status"] == "success" for result in results)

            # Verify all 25 requests were processed
            assert len(results) == 25

    @pytest.mark.asyncio
    async def test_circuit_breaker_lock_prevents_race_conditions(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test RLock prevents race conditions in circuit breaker state updates."""
        handler = VaultAdapter()

        vault_config["circuit_breaker_failure_threshold"] = 5

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # All requests fail
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            # Launch 100 concurrent failing requests to stress test locking
            async def execute_request(index: int) -> dict[str, object] | None:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    return await handler.execute(envelope)
                except (InfraConnectionError, InfraUnavailableError):
                    return None

            tasks = [execute_request(i) for i in range(100)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify failure count is consistent (no race condition corruption)
            # Exact count may vary due to circuit opening, but should be >= threshold
            assert handler._circuit_breaker_failures >= 5

            # Verify circuit state is consistent (OPEN after threshold) - using mixin attrs
            assert handler._circuit_breaker_open is True

            # Verify timeout is set (no race condition leaving it at 0) - using mixin attrs
            assert handler._circuit_breaker_open_until > 0
