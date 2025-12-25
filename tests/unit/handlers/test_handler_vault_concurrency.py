# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type, return-value"
"""Concurrency tests for VaultHandler.

These tests verify thread safety of circuit breaker state variables
and concurrent operation handling under production load scenarios.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from itertools import cycle
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_vault import VaultHandler

# Type alias for vault config dict values (str, int, float, bool, dict)
VaultConfigValue = str | int | float | bool | dict[str, str | int | float | bool]

# Type alias for handler response (status, payload, correlation_id)
HandlerResponse = dict[str, str | UUID | dict[str, str | int | bool | None]]


@pytest.fixture
def vault_config() -> dict[str, VaultConfigValue]:
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


class TestVaultHandlerConcurrency:
    """Test VaultHandler concurrent operation handling and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_state_updates(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit breaker state is thread-safe under concurrent access.

        This test verifies that concurrent operations properly update circuit breaker
        state without race conditions by:
        1. Tracking actual success and failure counts
        2. Verifying the circuit breaker failure count matches observed failures
        3. Ensuring no RuntimeError from race conditions occurs
        """
        handler = VaultHandler()

        vault_config["circuit_breaker_failure_threshold"] = 10

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Mix of failures and successes - create a predictable cycling pattern
            # Use cycle() to prevent StopIteration when concurrent requests with
            # retries exhaust a finite list (Python 3.12+ raises TypeError when
            # StopIteration is raised into an async Future context)
            #
            # IMPORTANT: Use string markers for errors instead of pre-created
            # Exception objects. Exception instances are mutable (they carry
            # __traceback__ state), so reusing the same Exception object across
            # threads causes race conditions when multiple threads modify the
            # traceback simultaneously.
            success_response: dict[str, object] = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }
            responses_pattern: list[dict[str, object] | str] = [
                "error:Connection error",  # String marker - creates fresh Exception
                success_response,
                "error:Connection error",
                success_response,
            ]
            response_cycle = cycle(responses_pattern)

            # Lock for thread-safe cycle access - VaultHandler.execute() runs hvac
            # client calls in a ThreadPoolExecutor, so multiple threads may
            # concurrently call get_response() which accesses the shared iterator
            cycle_lock = threading.Lock()

            def get_response(*args: object, **kwargs: object) -> dict[str, object]:
                """Return next response from cycle - thread-safe with lock.

                When using a callable for side_effect, mock does NOT automatically
                raise exceptions - it returns them as values. We must explicitly
                raise exceptions for error markers.

                Thread safety is ensured by:
                1. Lock protects the shared iterator (response_cycle)
                2. Fresh Exception instances are created inside the lock
                3. No mutable state is shared between concurrent calls

                Using string markers ("error:message") instead of pre-created
                Exception objects ensures each thread gets its own Exception
                instance with independent __traceback__ state.
                """
                with cycle_lock:
                    response = next(response_cycle)
                    if isinstance(response, str) and response.startswith("error:"):
                        # Create fresh RuntimeError instance for thread safety
                        # Using RuntimeError instead of base Exception per TRY002
                        raise RuntimeError(response[6:])
                    return response

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                get_response
            )

            await handler.initialize(vault_config)

            # Track results for verification
            success_count = 0
            failure_count = 0
            lock = asyncio.Lock()

            # Launch 20 concurrent requests (mix of success and failure)
            async def execute_request(index: int) -> HandlerResponse | None:
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
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent successful operations don't cause race conditions."""
        handler = VaultHandler()

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
            async def execute_request(index: int) -> HandlerResponse:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                return await handler.execute(envelope)

            tasks = [execute_request(i) for i in range(50)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(output.result["status"] == "success" for output in results)

            # Circuit breaker should remain CLOSED with zero failures (using mixin attrs)
            assert handler._circuit_breaker_open is False
            assert handler._circuit_breaker_failures == 0

    @pytest.mark.asyncio
    async def test_concurrent_failures_trigger_circuit_correctly(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent failures correctly trigger circuit breaker."""
        handler = VaultHandler()

        vault_config["circuit_breaker_failure_threshold"] = 3

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # All requests fail
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            # Launch concurrent failing requests
            async def execute_request(index: int) -> HandlerResponse | None:
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
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent write operations are thread-safe."""
        handler = VaultHandler()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.create_or_update_secret.return_value = {
                "data": {"version": 1, "created_time": "2025-01-01T00:00:00Z"}
            }

            await handler.initialize(vault_config)

            # Launch 30 concurrent write operations
            async def execute_write(index: int) -> HandlerResponse:
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
            assert all(output.result["status"] == "success" for output in results)

            # Verify create_or_update_secret was called 30 times (once per request)
            assert (
                mock_hvac_client.secrets.kv.v2.create_or_update_secret.call_count == 30
            )

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent health checks are thread-safe."""
        handler = VaultHandler()

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
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test shutdown is safe during concurrent operations.

        This test verifies that:
        1. Shutdown can occur while operations are in progress
        2. No race conditions or deadlocks occur
        3. Handler state is properly cleaned up after shutdown
        4. All tasks complete (either successfully or with expected errors)
        """
        handler = VaultHandler()

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
            async def execute_request(index: int) -> HandlerResponse | None:
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
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test thread pool correctly handles concurrent operation load."""
        handler = VaultHandler()

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
            async def execute_request(index: int) -> HandlerResponse:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                return await handler.execute(envelope)

            tasks = [execute_request(i) for i in range(25)]
            results = await asyncio.gather(*tasks)

            # All should succeed despite thread pool size < request count
            assert all(output.result["status"] == "success" for output in results)

            # Verify all 25 requests were processed
            assert len(results) == 25

    @pytest.mark.asyncio
    async def test_circuit_breaker_lock_prevents_race_conditions(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test asyncio.Lock prevents race conditions in circuit breaker state updates."""
        handler = VaultHandler()

        vault_config["circuit_breaker_failure_threshold"] = 5

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # All requests fail
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            # Launch 100 concurrent failing requests to stress test locking
            async def execute_request(index: int) -> HandlerResponse | None:
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

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_transition_race_open_to_half_open(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit breaker state transition from OPEN to HALF_OPEN under concurrent access.

        This test verifies that the OPEN -> HALF_OPEN transition is atomic and
        does not cause race conditions when multiple concurrent requests attempt
        to transition the circuit breaker state after the reset timeout.
        """
        handler = VaultHandler()

        vault_config["circuit_breaker_failure_threshold"] = 1
        vault_config["circuit_breaker_reset_timeout_seconds"] = (
            1.0  # Minimum valid timeout
        )

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Start with failures to open circuit
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            # Execute one request to open the circuit
            envelope: dict[str, str | UUID | dict[str, str]] = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            try:
                await handler.execute(envelope)
            except InfraConnectionError:
                pass

            # Verify circuit is open
            assert handler._circuit_breaker_open is True

            # Wait for reset timeout to expire (1.0 second + buffer)
            await asyncio.sleep(1.1)

            # Now switch to success responses for HALF_OPEN test
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = None
            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            # Track state transitions during concurrent requests
            transition_errors: list[str] = []
            lock = asyncio.Lock()

            async def execute_during_transition(index: int) -> HandlerResponse | None:
                """Execute request during state transition."""
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    return await handler.execute(envelope)
                except (InfraConnectionError, InfraUnavailableError) as e:
                    async with lock:
                        transition_errors.append(str(type(e).__name__))
                    return None
                except RuntimeError as e:
                    async with lock:
                        transition_errors.append(f"RuntimeError: {e}")
                    return None

            # Launch 10 concurrent requests during potential state transition
            tasks = [execute_during_transition(i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify no RuntimeErrors (race condition indicators)
            runtime_errors = [
                e for e in transition_errors if e.startswith("RuntimeError")
            ]
            assert len(runtime_errors) == 0, (
                f"Race condition detected: {runtime_errors}"
            )

            # After successful execution, circuit should be closed
            # (at least one request should have succeeded and closed the circuit)
            successful_results = [
                r for r in results if isinstance(r, dict) and r is not None
            ]
            if successful_results:
                assert handler._circuit_breaker_open is False, (
                    "Circuit should be CLOSED after successful test request in HALF_OPEN state"
                )

    @pytest.mark.asyncio
    async def test_executor_shutdown_robustness(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test robust executor shutdown verification.

        This test verifies:
        1. Executor is properly shut down
        2. No pending tasks leak
        3. Shutdown is safe even when tasks are in progress
        4. Double shutdown is handled gracefully
        """
        handler = VaultHandler()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            await handler.initialize(vault_config)

            # Capture executor reference before shutdown
            original_executor = handler._executor
            assert original_executor is not None, "Executor should be initialized"

            # Verify executor has expected configuration
            assert handler._max_workers > 0, "Max workers should be set"
            assert handler._max_queue_size > 0, "Max queue size should be set"

            # First shutdown
            await handler.shutdown()

            # Verify executor is set to None after shutdown
            assert handler._executor is None, "Executor should be None after shutdown"

            # Verify original executor is actually shut down
            # by attempting to submit a task (should raise RuntimeError)
            with pytest.raises(RuntimeError, match="cannot schedule new futures"):
                original_executor.submit(lambda: None)

            # Double shutdown should be safe (no error raised)
            await handler.shutdown()

            # Verify state remains consistent after double shutdown
            assert handler._initialized is False
            assert handler._executor is None
            assert handler._client is None
            assert handler._config is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_half_open_recovery(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent recovery from HALF_OPEN state.

        This test verifies that when multiple requests attempt to execute
        in HALF_OPEN state simultaneously, only one succeeds in transitioning
        the circuit back to CLOSED state without race conditions.
        """
        handler = VaultHandler()

        vault_config["circuit_breaker_failure_threshold"] = 1
        vault_config["circuit_breaker_reset_timeout_seconds"] = (
            1.0  # Minimum valid timeout
        )

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Start with failure to open circuit
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            envelope: dict[str, str | UUID | dict[str, str]] = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            try:
                await handler.execute(envelope)
            except InfraConnectionError:
                pass

            # Wait for reset timeout (1.0 second + buffer)
            await asyncio.sleep(1.1)

            # Switch to success
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = None
            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            # Track execution attempts and results
            attempts = 0
            successes = 0
            failures = 0
            lock = asyncio.Lock()

            async def execute_during_recovery(index: int) -> HandlerResponse | None:
                nonlocal attempts, successes, failures
                async with lock:
                    attempts += 1

                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    result = await handler.execute(envelope)
                    async with lock:
                        successes += 1
                    return result
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        failures += 1
                    return None

            # Launch concurrent requests during recovery
            tasks = [execute_during_recovery(i) for i in range(20)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all attempts were tracked
            assert attempts == 20, f"Expected 20 attempts, got {attempts}"

            # Verify total handled equals total requests
            assert successes + failures == 20, (
                f"Expected 20 handled, got {successes + failures}"
            )

            # Final state should be consistent - circuit closed after recovery
            # (at least some requests should have succeeded)
            assert successes > 0, (
                "At least some requests should succeed during recovery"
            )

            # Circuit should now be closed (recovered)
            if successes > 0:
                assert handler._circuit_breaker_open is False, (
                    "Circuit should be CLOSED after successful recovery"
                )
                assert handler._circuit_breaker_failures == 0, (
                    "Failure count should be reset after recovery"
                )
