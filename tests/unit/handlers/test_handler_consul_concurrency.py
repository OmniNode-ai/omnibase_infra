# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type, return-value"
"""Concurrency tests for HandlerConsul retry operations.

These tests verify that concurrent operations have isolated retry state
and no race conditions occur when multiple operations retry simultaneously.

The HandlerConsul was refactored to use explicit retry state passing
(via _init_retry_context and _try_execute_operation) rather than shared
instance variables, ensuring thread-safe concurrent retry behavior.
"""

from __future__ import annotations

import asyncio
import threading
from itertools import cycle
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import consul
import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.handlers.handler_consul import HandlerConsul

# Type alias for consul config dict values
ConsulConfigValue = str | int | float | bool | dict[str, str | int | float | bool]

# Type alias for handler response
HandlerResponse = dict[str, str | UUID | dict[str, str | int | bool | None]]


@pytest.fixture
def consul_config() -> dict[str, ConsulConfigValue]:
    """Provide test Consul configuration with retry settings."""
    return {
        "host": "consul.example.com",
        "port": 8500,
        "scheme": "http",
        "token": "acl-token-abc123",
        "timeout_seconds": 30.0,
        "connect_timeout_seconds": 10.0,
        "datacenter": "dc1",
        "health_check_interval_seconds": 30.0,
        "max_concurrent_operations": 20,
        "retry": {
            "max_attempts": 3,
            "initial_delay_seconds": 0.1,  # Minimum allowed
            "max_delay_seconds": 1.0,  # Minimum allowed
            "exponential_base": 2.0,
        },
    }


@pytest.fixture
def mock_consul_client() -> MagicMock:
    """Provide mocked consul.Consul client."""
    client = MagicMock()

    # Mock Status operations (for health check during initialization)
    client.status = MagicMock()
    client.status.leader = MagicMock(return_value="192.168.1.1:8300")

    # Mock KV store operations
    client.kv = MagicMock()
    client.kv.get = MagicMock(
        return_value=(
            0,
            {"Value": b"test-value", "Key": "test/key", "ModifyIndex": 100},
        )
    )
    client.kv.put = MagicMock(return_value=True)

    # Mock Agent operations
    client.agent = MagicMock()
    client.agent.service = MagicMock()
    client.agent.service.register = MagicMock(return_value=None)
    client.agent.service.deregister = MagicMock(return_value=None)

    return client


class TestHandlerConsulConcurrentRetry:
    """Test HandlerConsul concurrent retry operation handling."""

    @pytest.mark.asyncio
    async def test_concurrent_retry_operations_isolated_state(
        self,
        consul_config: dict[str, ConsulConfigValue],
        mock_consul_client: MagicMock,
    ) -> None:
        """Verify concurrent operations have isolated retry state.

        This test ensures that when multiple operations fail and retry
        simultaneously, each maintains its own independent retry count
        and backoff state. No shared mutable state should cause interference.
        """
        handler = HandlerConsul()

        # Use higher retry threshold to allow more concurrent failures
        consul_config["circuit_breaker_failure_threshold"] = 20  # Max allowed

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Create a pattern where first attempt fails, second succeeds per op
            # Use cycling pattern with thread-safe access
            success_response = (
                0,
                {"Value": b"test-value", "Key": "test/key", "ModifyIndex": 100},
            )

            # Each operation needs: fail, succeed (2 attempts due to retry)
            # With 10 concurrent operations, we need at least 20 responses
            responses_pattern: list[tuple[int, dict[str, object] | None] | str] = [
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

            def get_response(
                *args: object, **kwargs: object
            ) -> tuple[int, dict[str, object] | None]:
                """Return next response from cycle - thread-safe with lock."""
                with cycle_lock:
                    response = next(response_cycle)
                    if isinstance(response, str) and response.startswith("error:"):
                        # Create fresh exception for thread safety
                        raise consul.ConsulException(response[6:])
                    return response

            mock_consul_client.kv.get.side_effect = get_response

            await handler.initialize(consul_config)

            # Track results for verification
            success_count = 0
            failure_count = 0
            retry_attempt_counts: list[int] = []
            lock = asyncio.Lock()

            async def execute_request(index: int) -> HandlerResponse | None:
                nonlocal success_count, failure_count
                envelope = {
                    "operation": "consul.kv_get",
                    "payload": {"key": f"test/key{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    result = await handler.execute(envelope)
                    async with lock:
                        success_count += 1
                    return result
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
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
        consul_config: dict[str, ConsulConfigValue],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test mixed success/failure operations don't interfere with retry state.

        Some operations succeed immediately, others fail and retry.
        Each operation's retry logic should be independent.
        """
        handler = HandlerConsul()

        consul_config["circuit_breaker_failure_threshold"] = 20  # Max allowed

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Thread-safe counter for deterministic behavior
            call_counter = 0
            counter_lock = threading.Lock()

            success_response = (
                0,
                {"Value": b"test-value", "Key": "test/key", "ModifyIndex": 100},
            )

            def get_response(
                *args: object, **kwargs: object
            ) -> tuple[int, dict[str, object] | None]:
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
                    raise consul.ConsulException("Transient error")
                return success_response

            mock_consul_client.kv.get.side_effect = get_response

            await handler.initialize(consul_config)

            # Track per-operation success
            operation_results: dict[int, bool] = {}
            lock = asyncio.Lock()

            async def execute_request(index: int) -> HandlerResponse | None:
                envelope = {
                    "operation": "consul.kv_get",
                    "payload": {"key": f"test/key{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    result = await handler.execute(envelope)
                    async with lock:
                        operation_results[index] = True
                    return result
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
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
        consul_config: dict[str, ConsulConfigValue],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test all concurrent operations retrying to exhaustion independently.

        When all operations fail repeatedly, each should track its own
        retry count and all should exhaust retries independently.
        """
        handler = HandlerConsul()

        # Set threshold high enough for concurrent testing
        consul_config["circuit_breaker_failure_threshold"] = 20  # Max allowed
        # Set retry attempts to 2 for faster test
        consul_config["retry"]["max_attempts"] = 2

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # All requests fail
            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Persistent error"
            )

            await handler.initialize(consul_config)

            # Track retry behavior
            failure_count = 0
            lock = asyncio.Lock()

            async def execute_request(index: int) -> HandlerResponse | None:
                nonlocal failure_count
                envelope = {
                    "operation": "consul.kv_get",
                    "payload": {"key": f"test/key{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    return await handler.execute(envelope)
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
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
        consul_config: dict[str, ConsulConfigValue],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test concurrent retry across different operation types.

        Multiple operation types (kv_get, kv_put, register) running
        concurrently should each have isolated retry state.
        """
        handler = HandlerConsul()

        consul_config["circuit_breaker_failure_threshold"] = 20  # Max allowed

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Set up per-operation behavior
            kv_get_calls = 0
            kv_put_calls = 0
            register_calls = 0
            calls_lock = threading.Lock()

            def kv_get_response(
                *args: object, **kwargs: object
            ) -> tuple[int, dict[str, object] | None]:
                nonlocal kv_get_calls
                with calls_lock:
                    kv_get_calls += 1
                    current = kv_get_calls
                # Fail first call, succeed second
                if current == 1:
                    raise consul.ConsulException("KV get transient error")
                return (0, {"Value": b"test", "Key": "test/key", "ModifyIndex": 1})

            def kv_put_response(*args: object, **kwargs: object) -> bool:
                nonlocal kv_put_calls
                with calls_lock:
                    kv_put_calls += 1
                    current = kv_put_calls
                # Fail first call, succeed second
                if current == 1:
                    raise consul.ConsulException("KV put transient error")
                return True

            def register_response(*args: object, **kwargs: object) -> None:
                nonlocal register_calls
                with calls_lock:
                    register_calls += 1
                    current = register_calls
                # Fail first call, succeed second
                if current == 1:
                    raise consul.ConsulException("Register transient error")

            mock_consul_client.kv.get.side_effect = kv_get_response
            mock_consul_client.kv.put.side_effect = kv_put_response
            mock_consul_client.agent.service.register.side_effect = register_response

            await handler.initialize(consul_config)

            # Track per-operation-type results
            results: dict[str, bool] = {}
            lock = asyncio.Lock()

            async def execute_kv_get() -> None:
                envelope = {
                    "operation": "consul.kv_get",
                    "payload": {"key": "test/key"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        results["kv_get"] = True
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
                    async with lock:
                        results["kv_get"] = False

            async def execute_kv_put() -> None:
                envelope = {
                    "operation": "consul.kv_put",
                    "payload": {"key": "test/key", "value": "test"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        results["kv_put"] = True
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
                    async with lock:
                        results["kv_put"] = False

            async def execute_register() -> None:
                envelope = {
                    "operation": "consul.register",
                    "payload": {"name": "test-service"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        results["register"] = True
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
                    async with lock:
                        results["register"] = False

            # Launch all operation types concurrently
            await asyncio.gather(
                execute_kv_get(),
                execute_kv_put(),
                execute_register(),
            )

            # All operations should have succeeded (after retry)
            assert results.get("kv_get") is True, "kv_get should succeed after retry"
            assert results.get("kv_put") is True, "kv_put should succeed after retry"
            assert results.get("register") is True, (
                "register should succeed after retry"
            )

            # Verify each operation type was called multiple times (retry happened)
            assert kv_get_calls >= 2, f"Expected >= 2 kv_get calls, got {kv_get_calls}"
            assert kv_put_calls >= 2, f"Expected >= 2 kv_put calls, got {kv_put_calls}"
            assert register_calls >= 2, (
                f"Expected >= 2 register calls, got {register_calls}"
            )

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_retry_no_state_leakage_between_handlers(
        self,
        consul_config: dict[str, ConsulConfigValue],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that multiple handler instances have completely isolated retry state.

        Even when using multiple handler instances concurrently, each
        should maintain completely independent retry state.
        """
        consul_config["circuit_breaker_failure_threshold"] = 20  # Max allowed

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Create counter per handler instance (simulate different behavior)
            handler1_calls = 0
            handler2_calls = 0
            lock = threading.Lock()

            success_response = (
                0,
                {"Value": b"test-value", "Key": "test/key", "ModifyIndex": 100},
            )

            def get_response_handler1(
                *args: object, **kwargs: object
            ) -> tuple[int, dict[str, object] | None]:
                nonlocal handler1_calls
                with lock:
                    handler1_calls += 1
                    current = handler1_calls
                # Handler 1: fail first 2 calls, then succeed
                if current <= 2:
                    raise consul.ConsulException("Handler1 error")
                return success_response

            def get_response_handler2(
                *args: object, **kwargs: object
            ) -> tuple[int, dict[str, object] | None]:
                nonlocal handler2_calls
                with lock:
                    handler2_calls += 1
                # Handler 2: always succeed
                return success_response

            # Create two handlers
            handler1 = HandlerConsul()
            handler2 = HandlerConsul()

            mock_consul_client.kv.get.side_effect = get_response_handler1
            await handler1.initialize(consul_config)

            # Re-patch for handler2
            mock_consul_client.kv.get.side_effect = get_response_handler2
            await handler2.initialize(consul_config)

            # Now set up per-handler response behavior using key-based differentiation
            # This allows us to test that each handler has isolated retry state
            key_call_counts: dict[str, int] = {}

            def combined_response(
                key: str, *args: object, **kwargs: object
            ) -> tuple[int, dict[str, object] | None]:
                """Return different responses based on which handler is calling.

                Handler1 (h1/ keys): Fail first call per key, succeed on retry
                Handler2 (h2/ keys): Always succeed immediately
                """
                with lock:
                    if key.startswith("h1/"):
                        # Track calls per unique key for handler1
                        key_call_counts[key] = key_call_counts.get(key, 0) + 1
                        # Fail first call for each unique key to force retry
                        if key_call_counts[key] == 1:
                            raise consul.ConsulException("Handler1 execution error")
                        return success_response
                    else:  # h2/ keys
                        # Handler2 always succeeds immediately
                        key_call_counts[key] = key_call_counts.get(key, 0) + 1
                        return success_response

            mock_consul_client.kv.get.side_effect = combined_response

            # Execute on both handlers concurrently
            results1: list[bool] = []
            results2: list[bool] = []
            lock2 = asyncio.Lock()

            async def execute_on_handler1(index: int) -> None:
                envelope = {
                    "operation": "consul.kv_get",
                    "payload": {"key": f"h1/key{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler1.execute(envelope)
                    async with lock2:
                        results1.append(True)
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
                    async with lock2:
                        results1.append(False)

            async def execute_on_handler2(index: int) -> None:
                envelope = {
                    "operation": "consul.kv_get",
                    "payload": {"key": f"h2/key{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler2.execute(envelope)
                    async with lock2:
                        results2.append(True)
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
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
            # Each of 5 h1/ keys should have been called exactly 2 times
            h1_keys = [k for k in key_call_counts if k.startswith("h1/")]
            assert len(h1_keys) == 5, f"Expected 5 h1/ keys, got {len(h1_keys)}"
            for key in h1_keys:
                assert key_call_counts[key] == 2, (
                    f"Expected 2 calls for {key} (fail + retry), got {key_call_counts[key]}"
                )

            # Verify handler2 succeeded immediately without retry
            # Each of 5 h2/ keys should have been called exactly 1 time
            h2_keys = [k for k in key_call_counts if k.startswith("h2/")]
            assert len(h2_keys) == 5, f"Expected 5 h2/ keys, got {len(h2_keys)}"
            for key in h2_keys:
                assert key_call_counts[key] == 1, (
                    f"Expected 1 call for {key} (immediate success), got {key_call_counts[key]}"
                )

            # Verify handlers are still independent
            assert handler1._initialized is True
            assert handler2._initialized is True
            # Both handlers have independent clients (may be same mock, but state isolated)
            assert handler1._client is not None
            assert handler2._client is not None

            await handler1.shutdown()
            await handler2.shutdown()

    @pytest.mark.asyncio
    async def test_high_concurrency_stress_test(
        self,
        consul_config: dict[str, ConsulConfigValue],
        mock_consul_client: MagicMock,
    ) -> None:
        """Stress test with high concurrency to detect race conditions.

        Launch many concurrent operations with mixed success/failure
        to stress test the retry state isolation under load.
        """
        handler = HandlerConsul()

        consul_config["circuit_breaker_failure_threshold"] = 20  # Max allowed
        consul_config["max_concurrent_operations"] = 50

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Use cycling pattern for predictable but varied behavior
            success_response = (
                0,
                {"Value": b"test-value", "Key": "test/key", "ModifyIndex": 100},
            )
            responses_pattern: list[tuple[int, dict[str, object] | None] | str] = [
                success_response,
                success_response,
                "error:Transient",
                success_response,
            ]
            response_cycle = cycle(responses_pattern)
            cycle_lock = threading.Lock()

            def get_response(
                *args: object, **kwargs: object
            ) -> tuple[int, dict[str, object] | None]:
                with cycle_lock:
                    response = next(response_cycle)
                    if isinstance(response, str) and response.startswith("error:"):
                        raise consul.ConsulException(response[6:])
                    return response

            mock_consul_client.kv.get.side_effect = get_response

            await handler.initialize(consul_config)

            # Track results
            success_count = 0
            failure_count = 0
            race_errors: list[str] = []
            lock = asyncio.Lock()

            async def execute_request(index: int) -> None:
                nonlocal success_count, failure_count
                envelope = {
                    "operation": "consul.kv_get",
                    "payload": {"key": f"stress/key{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        success_count += 1
                except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError):
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
    async def test_concurrent_timeout_retry_isolation(
        self,
        consul_config: dict[str, ConsulConfigValue],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test concurrent timeout scenarios have isolated retry state.

        Multiple operations timing out should each track their own
        retry attempts independently.
        """
        handler = HandlerConsul()

        consul_config["circuit_breaker_failure_threshold"] = 20  # Max allowed
        consul_config["timeout_seconds"] = 1.0  # Minimum allowed
        consul_config["retry"]["max_attempts"] = 2

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Simulate timeouts with immediate success after
            timeout_count = 0
            timeout_lock = threading.Lock()

            def get_response(
                *args: object, **kwargs: object
            ) -> tuple[int, dict[str, object] | None]:
                nonlocal timeout_count
                with timeout_lock:
                    timeout_count += 1
                    current = timeout_count
                # First 5 calls timeout, rest succeed
                if current <= 5:
                    raise consul.Timeout("Operation timed out")
                return (0, {"Value": b"test", "Key": "key", "ModifyIndex": 1})

            mock_consul_client.kv.get.side_effect = get_response

            await handler.initialize(consul_config)

            success_count = 0
            timeout_failure_count = 0
            lock = asyncio.Lock()

            async def execute_request(index: int) -> None:
                nonlocal success_count, timeout_failure_count
                envelope = {
                    "operation": "consul.kv_get",
                    "payload": {"key": f"timeout/key{index}"},
                    "correlation_id": uuid4(),
                }
                try:
                    await handler.execute(envelope)
                    async with lock:
                        success_count += 1
                except InfraTimeoutError:
                    async with lock:
                        timeout_failure_count += 1
                except (InfraConnectionError, InfraUnavailableError):
                    async with lock:
                        timeout_failure_count += 1

            # Launch 10 concurrent requests
            tasks = [execute_request(i) for i in range(10)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all requests were processed
            total = success_count + timeout_failure_count
            assert total == 10, f"Expected 10 processed, got {total}"

            await handler.shutdown()


__all__: list[str] = [
    "TestHandlerConsulConcurrentRetry",
]
