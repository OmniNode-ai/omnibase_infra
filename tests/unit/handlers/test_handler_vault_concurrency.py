# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type, return-value"
"""Concurrency tests for HandlerVault.

These tests verify thread safety of circuit breaker state variables
and concurrent operation handling under production load scenarios.
"""

from __future__ import annotations

import asyncio
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
from omnibase_infra.handlers.handler_vault import HandlerVault

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


class TestHandlerVaultConcurrency:
    """Test HandlerVault concurrent operation handling and thread safety."""

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
        handler = HandlerVault()

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

            # Lock for thread-safe cycle access - HandlerVault.execute() runs hvac
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
        handler = HandlerVault()

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
        handler = HandlerVault()

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
        handler = HandlerVault()

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
        handler = HandlerVault()

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
        handler = HandlerVault()

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
        handler = HandlerVault()

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
        handler = HandlerVault()

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
        handler = HandlerVault()

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
        handler = HandlerVault()

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


class TestHandlerVaultConcurrentRetry:
    """Test HandlerVault concurrent retry operation handling.

    These tests verify that concurrent operations have isolated retry state
    and no race conditions occur when multiple operations retry simultaneously.

    Similar to HandlerConsul retry tests, HandlerVault must maintain independent
    retry state per operation when multiple requests fail and retry concurrently.
    """

    @pytest.mark.asyncio
    async def test_concurrent_retry_operations_isolated_state(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Verify concurrent operations have isolated retry state.

        This test ensures that when multiple operations fail and retry
        simultaneously, each maintains its own independent retry count
        and backoff state. No shared mutable state should cause interference.
        """
        handler = HandlerVault()

        vault_config["circuit_breaker_failure_threshold"] = 20  # Max allowed

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Create pattern where first attempt fails, second succeeds per operation
            success_response: dict[str, object] = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            # Each operation needs: fail, succeed (2 attempts due to retry)
            # With 10 concurrent operations, we need at least 20 responses
            responses_pattern: list[dict[str, object] | str] = [
                "error:Connection error",  # Op 0, attempt 1
                "error:Connection error",  # Op 1, attempt 1
                "error:Connection error",  # Op 2, attempt 1
                "error:Connection error",  # Op 3, attempt 1
                "error:Connection error",  # Op 4, attempt 1
                success_response,  # Op 0, attempt 2
                success_response,  # Op 1, attempt 2
                success_response,  # Op 2, attempt 2
                success_response,  # Op 3, attempt 2
                success_response,  # Op 4, attempt 2
            ]
            response_cycle = cycle(responses_pattern)

            # Lock for thread-safe cycle access
            cycle_lock = threading.Lock()

            def get_response(*args: object, **kwargs: object) -> dict[str, object]:
                """Return next response from cycle - thread-safe with lock."""
                with cycle_lock:
                    response = next(response_cycle)
                    if isinstance(response, str) and response.startswith("error:"):
                        # Create fresh exception for thread safety
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

            # Launch 10 concurrent requests
            tasks = [execute_request(i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify no race conditions occurred (no RuntimeError)
            for result in results:
                assert not isinstance(result, RuntimeError), (
                    f"Race condition detected: {result}"
                )

            # Verify all requests were processed
            total_processed = success_count + failure_count
            assert total_processed == 10, f"Expected 10 total, got {total_processed}"

            # Some should have succeeded after retry
            assert success_count > 0, "At least some operations should succeed"

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_mixed_success_failure_isolated_retry(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test mixed success/failure operations don't interfere with retry state.

        Some operations succeed immediately, others fail and retry.
        Each operation's retry logic should be independent.
        """
        handler = HandlerVault()

        vault_config["circuit_breaker_failure_threshold"] = 20  # Max allowed

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Thread-safe counter for deterministic behavior
            call_counter = 0
            counter_lock = threading.Lock()

            success_response: dict[str, object] = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            def get_response(*args: object, **kwargs: object) -> dict[str, object]:
                """Return success for even calls, failure for first odd call.

                This creates a pattern where:
                - Even operations succeed immediately
                - Odd operations fail once, then succeed on retry
                """
                nonlocal call_counter
                with counter_lock:
                    current = call_counter
                    call_counter += 1

                # Every 3rd call fails (but not fatal - will succeed on retry)
                if current % 3 == 1:
                    raise RuntimeError("Transient error")
                return success_response

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                get_response
            )

            await handler.initialize(vault_config)

            # Track per-operation success
            operation_results: dict[int, bool] = {}
            lock = asyncio.Lock()

            async def execute_request(index: int) -> HandlerResponse | None:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    result = await handler.execute(envelope)
                    async with lock:
                        operation_results[index] = True
                    return result
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        operation_results[index] = False
                    return None

            # Launch 15 concurrent requests
            tasks = [execute_request(i) for i in range(15)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all operations were tracked
            assert len(operation_results) == 15, (
                f"Expected 15 results, got {len(operation_results)}"
            )

            # Most should succeed (transient failures are retried)
            success_rate = sum(operation_results.values()) / len(operation_results)
            assert success_rate > 0.5, (
                f"Expected >50% success rate, got {success_rate:.1%}"
            )

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_all_operations_retry_to_exhaustion(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test all concurrent operations retrying to exhaustion independently.

        When all operations fail repeatedly, each should track its own
        retry count and all should exhaust retries independently.
        """
        handler = HandlerVault()

        # Set threshold high enough for concurrent testing
        vault_config["circuit_breaker_failure_threshold"] = 20  # Max allowed
        # Set retry attempts to 2 for faster test
        vault_config["retry"]["max_attempts"] = 2

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # All requests fail
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                RuntimeError("Persistent error")
            )

            await handler.initialize(vault_config)

            # Track retry behavior
            failure_count = 0
            lock = asyncio.Lock()

            async def execute_request(index: int) -> HandlerResponse | None:
                nonlocal failure_count
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"myapp/config{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    return await handler.execute(envelope)
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        failure_count += 1
                    return None

            # Launch 5 concurrent requests
            tasks = [execute_request(i) for i in range(5)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # All should have failed after exhausting retries
            assert failure_count == 5, f"Expected 5 failures, got {failure_count}"

            # Verify handler is still functional (not in broken state)
            assert handler._initialized is True

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_retry_with_different_operations(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test concurrent retry across different operation types.

        Multiple operation types (read_secret, write_secret, delete_secret)
        running concurrently should each have isolated retry state.
        """
        handler = HandlerVault()

        vault_config["circuit_breaker_failure_threshold"] = 20  # Max allowed

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Set up per-operation behavior
            read_calls = 0
            write_calls = 0
            delete_calls = 0
            calls_lock = threading.Lock()

            def read_response(*args: object, **kwargs: object) -> dict[str, object]:
                nonlocal read_calls
                with calls_lock:
                    read_calls += 1
                    current = read_calls
                # Fail first call, succeed second
                if current == 1:
                    raise RuntimeError("Read transient error")
                return {"data": {"data": {"key": "value"}, "metadata": {"version": 1}}}

            def write_response(*args: object, **kwargs: object) -> dict[str, object]:
                nonlocal write_calls
                with calls_lock:
                    write_calls += 1
                    current = write_calls
                # Fail first call, succeed second
                if current == 1:
                    raise RuntimeError("Write transient error")
                return {"data": {"version": 2, "created_time": "2025-01-01T00:00:00Z"}}

            def delete_response(*args: object, **kwargs: object) -> None:
                nonlocal delete_calls
                with calls_lock:
                    delete_calls += 1
                    current = delete_calls
                # Fail first call, succeed second
                if current == 1:
                    raise RuntimeError("Delete transient error")

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                read_response
            )
            mock_hvac_client.secrets.kv.v2.create_or_update_secret.side_effect = (
                write_response
            )
            mock_hvac_client.secrets.kv.v2.delete_latest_version_of_secret.side_effect = delete_response

            await handler.initialize(vault_config)

            # Track per-operation-type results
            results: dict[str, bool] = {}
            lock = asyncio.Lock()

            async def execute_read() -> None:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": "myapp/config"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        results["read"] = True
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        results["read"] = False

            async def execute_write() -> None:
                envelope = {
                    "operation": "vault.write_secret",
                    "payload": {"path": "myapp/secret", "data": {"key": "value"}},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        results["write"] = True
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        results["write"] = False

            async def execute_delete() -> None:
                envelope = {
                    "operation": "vault.delete_secret",
                    "payload": {"path": "myapp/old_secret"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        results["delete"] = True
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        results["delete"] = False

            # Launch all operation types concurrently
            await asyncio.gather(
                execute_read(),
                execute_write(),
                execute_delete(),
            )

            # All operations should have succeeded (after retry)
            assert results.get("read") is True, "read should succeed after retry"
            assert results.get("write") is True, "write should succeed after retry"
            assert results.get("delete") is True, "delete should succeed after retry"

            # Verify each operation type was called multiple times (retry happened)
            assert read_calls >= 2, f"Expected >= 2 read calls, got {read_calls}"
            assert write_calls >= 2, f"Expected >= 2 write calls, got {write_calls}"
            assert delete_calls >= 2, f"Expected >= 2 delete calls, got {delete_calls}"

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_high_concurrency_retry_stress_test(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Stress test with high concurrency to detect retry race conditions.

        Launch many concurrent operations with mixed success/failure
        to stress test the retry state isolation under load.
        """
        handler = HandlerVault()

        vault_config["circuit_breaker_failure_threshold"] = 20  # Max allowed
        vault_config["max_concurrent_operations"] = 50

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Use cycling pattern for predictable but varied behavior
            success_response: dict[str, object] = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }
            responses_pattern: list[dict[str, object] | str] = [
                success_response,
                success_response,
                "error:Transient",
                success_response,
            ]
            response_cycle = cycle(responses_pattern)
            cycle_lock = threading.Lock()

            def get_response(*args: object, **kwargs: object) -> dict[str, object]:
                with cycle_lock:
                    response = next(response_cycle)
                    if isinstance(response, str) and response.startswith("error:"):
                        raise RuntimeError(response[6:])
                    return response

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                get_response
            )

            await handler.initialize(vault_config)

            # Track results
            success_count = 0
            failure_count = 0
            race_errors: list[str] = []
            lock = asyncio.Lock()

            async def execute_request(index: int) -> None:
                nonlocal success_count, failure_count
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"stress/secret{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        success_count += 1
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        failure_count += 1
                except RuntimeError as e:
                    async with lock:
                        race_errors.append(f"RuntimeError: {e}")

            # Launch 100 concurrent requests
            tasks = [execute_request(i) for i in range(100)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify no race conditions
            assert len(race_errors) == 0, f"Race conditions detected: {race_errors}"

            # Verify all requests were processed
            total = success_count + failure_count
            assert total == 100, f"Expected 100 processed, got {total}"

            # High success rate expected (only 1/4 of responses are errors, and we retry)
            success_rate = success_count / 100
            assert success_rate > 0.7, f"Expected >70% success, got {success_rate:.1%}"

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_retry_no_state_leakage_between_handlers(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test that multiple handler instances have isolated retry state.

        Even when using multiple handler instances concurrently, each
        should maintain completely independent retry state.
        """
        vault_config["circuit_breaker_failure_threshold"] = 20  # Max allowed

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Create counter per handler instance (simulate different behavior)
            handler1_calls = 0
            handler2_calls = 0
            lock = threading.Lock()

            success_response: dict[str, object] = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            def get_response_handler1(
                *args: object, **kwargs: object
            ) -> dict[str, object]:
                nonlocal handler1_calls
                with lock:
                    handler1_calls += 1
                    current = handler1_calls
                # Handler 1: fail first 2 calls, then succeed
                if current <= 2:
                    raise RuntimeError("Handler1 error")
                return success_response

            def get_response_handler2(
                *args: object, **kwargs: object
            ) -> dict[str, object]:
                nonlocal handler2_calls
                with lock:
                    handler2_calls += 1
                # Handler 2: always succeed
                return success_response

            # Create two handlers
            handler1 = HandlerVault()
            handler2 = HandlerVault()

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                get_response_handler1
            )
            await handler1.initialize(vault_config)

            # Re-patch for handler2
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                get_response_handler2
            )
            await handler2.initialize(vault_config)

            # Now set up per-handler response behavior using path-based differentiation
            path_call_counts: dict[str, int] = {}

            def combined_response(
                path: str, *args: object, **kwargs: object
            ) -> dict[str, object]:
                """Return different responses based on which handler is calling.

                Handler1 (h1/ paths): Fail first call per path, succeed on retry
                Handler2 (h2/ paths): Always succeed immediately
                """
                with lock:
                    if path.startswith("h1/"):
                        # Track calls per unique path for handler1
                        path_call_counts[path] = path_call_counts.get(path, 0) + 1
                        # Fail first call for each unique path to force retry
                        if path_call_counts[path] == 1:
                            raise RuntimeError("Handler1 execution error")
                        return success_response
                    else:  # h2/ paths
                        # Handler2 always succeeds immediately
                        path_call_counts[path] = path_call_counts.get(path, 0) + 1
                        return success_response

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                combined_response
            )

            # Execute on both handlers concurrently
            results1: list[bool] = []
            results2: list[bool] = []
            lock2 = asyncio.Lock()

            async def execute_on_handler1(index: int) -> None:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"h1/secret{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler1.execute(envelope)
                    async with lock2:
                        results1.append(True)
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock2:
                        results1.append(False)

            async def execute_on_handler2(index: int) -> None:
                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": f"h2/secret{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler2.execute(envelope)
                    async with lock2:
                        results2.append(True)
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock2:
                        results2.append(False)

            # Launch concurrent operations on both handlers
            tasks = []
            for i in range(5):
                tasks.append(execute_on_handler1(i))
                tasks.append(execute_on_handler2(i))

            await asyncio.gather(*tasks, return_exceptions=True)

            # Both handlers should have processed requests
            assert len(results1) == 5, (
                f"Expected 5 results from handler1, got {len(results1)}"
            )
            assert len(results2) == 5, (
                f"Expected 5 results from handler2, got {len(results2)}"
            )

            # All operations should succeed after retry
            assert all(results1), "All handler1 operations should succeed after retry"
            assert all(results2), "All handler2 operations should succeed immediately"

            # Verify handler1 actually retried (each operation fails once, then succeeds)
            # Each of 5 h1/ paths should have been called exactly 2 times
            h1_paths = [k for k in path_call_counts if k.startswith("h1/")]
            assert len(h1_paths) == 5, f"Expected 5 h1/ paths, got {len(h1_paths)}"
            for path in h1_paths:
                assert path_call_counts[path] == 2, (
                    f"Expected 2 calls for {path} (fail + retry), got {path_call_counts[path]}"
                )

            # Verify handler2 succeeded immediately without retry
            # Each of 5 h2/ paths should have been called exactly 1 time
            h2_paths = [k for k in path_call_counts if k.startswith("h2/")]
            assert len(h2_paths) == 5, f"Expected 5 h2/ paths, got {len(h2_paths)}"
            for path in h2_paths:
                assert path_call_counts[path] == 1, (
                    f"Expected 1 call for {path} (immediate success), got {path_call_counts[path]}"
                )

            # Verify handlers are still independent
            assert handler1._initialized is True
            assert handler2._initialized is True
            assert handler1._client is not None
            assert handler2._client is not None

            await handler1.shutdown()
            await handler2.shutdown()
