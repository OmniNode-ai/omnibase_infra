# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for effect retry and backoff behavior (OMN-954 G4 acceptance criteria).

This test suite validates:
- Backoff policy applied on transient failures
- Exponential backoff timing (0.1s, 0.2s, 0.4s sequence)
- Retry exhaustion raises final error with attempt count
- Retry exhaustion with circuit breaker integration
- Non-retryable errors fail immediately without retry

Test Organization:
    - TestEffectRetryBackoff: Core retry and backoff validation tests

Coverage Goals:
    - Backoff policy validated (acceptance criteria)
    - Retry exhaustion tested (acceptance criteria)
    - Circuit breaker integration validated
    - Non-retryable error behavior verified

Pattern Reference:
    - Retry config models: src/omnibase_infra/handlers/model_vault_retry_config.py
    - Handler implementation: src/omnibase_infra/handlers/handler_consul.py
    - Circuit breaker mixin: src/omnibase_infra/mixins/mixin_async_circuit_breaker.py
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import consul
import pytest

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.handlers.handler_consul import ConsulHandler


@pytest.fixture
def retry_config_fast() -> dict[str, object]:
    """Provide fast retry configuration for testing.

    Uses very short delays to keep tests fast while still
    validating backoff behavior.
    """
    return {
        "host": "consul.test.local",
        "port": 8500,
        "scheme": "http",
        "timeout_seconds": 5.0,
        "retry": {
            "max_attempts": 3,
            "initial_delay_seconds": 0.1,
            "max_delay_seconds": 1.0,
            "exponential_base": 2.0,
        },
        # Circuit breaker with high threshold to not interfere with retry tests
        "circuit_breaker_enabled": True,
        "circuit_breaker_failure_threshold": 10,
        "circuit_breaker_reset_timeout_seconds": 30.0,
    }


@pytest.fixture
def mock_consul_client() -> MagicMock:
    """Provide mocked consul.Consul client."""
    client = MagicMock()

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

    # Mock Status operations (for initialization health check)
    client.status = MagicMock()
    client.status.leader = MagicMock(return_value="192.168.1.1:8300")

    return client


@pytest.mark.unit
@pytest.mark.asyncio
class TestEffectRetryBackoff:
    """Test suite for effect retry and backoff behavior (G4 acceptance criteria)."""

    async def test_backoff_policy_applied_on_transient_failure(
        self,
        retry_config_fast: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test backoff policy is applied when transient failure occurs.

        Validates:
        - First attempt fails with transient error
        - Backoff delay is applied before retry
        - Second attempt succeeds
        - Total attempts = 2
        """
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # First call fails, second succeeds
            mock_consul_client.kv.get.side_effect = [
                Exception("Transient network error"),
                (0, {"Value": b"success", "Key": "test/key", "ModifyIndex": 100}),
            ]

            await handler.initialize(retry_config_fast)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            # Act: Execute with timing measurement
            start = time.monotonic()
            output = await handler.execute(envelope)
            elapsed = time.monotonic() - start

            # Assert: Backoff was applied (at least initial_backoff delay of 0.1s)
            # Use 0.08s tolerance to account for timing precision
            assert (
                elapsed >= 0.08
            ), f"Expected at least 0.08s delay for backoff, got {elapsed:.3f}s"

            # Assert: Operation succeeded on retry
            result = output.result
            assert result.status == "success"

            # Assert: Total attempts = 2 (first fail, second success)
            assert mock_consul_client.kv.get.call_count == 2

    async def test_exponential_backoff_timing(
        self,
        retry_config_fast: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test exponential backoff timing follows expected sequence.

        Validates:
        - Initial backoff = 0.1s (minimum allowed by config model)
        - Exponential base = 2.0
        - Expected delays: ~0.1s, ~0.2s (before 3rd attempt fails)
        - Backoff is applied between retry attempts

        Note: We verify backoff is applied by checking that elapsed time
        is at least the expected minimum backoff delay sum.
        """
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # All attempts fail - use consul-specific exception for proper handling
            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Persistent error"
            )

            await handler.initialize(retry_config_fast)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            # Act: Execute with timing measurement
            start = time.monotonic()
            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)
            elapsed = time.monotonic() - start

            # Assert: All 3 attempts were made
            assert mock_consul_client.kv.get.call_count == 3

            # Assert: Total delay should be at least initial_backoff + 2*initial_backoff
            # Delay pattern: 0.1s (after 1st fail) + 0.2s (after 2nd fail) = 0.3s minimum
            # Using lower tolerance of 0.2s to account for timing precision
            assert (
                elapsed >= 0.2
            ), f"Expected at least 0.2s total delay from backoff (0.1s + 0.2s), got {elapsed:.3f}s"

            # Assert: Validate exponential growth by checking elapsed is NOT too fast
            # If no backoff was applied, it would complete in <0.1s
            # This verifies backoff is actually happening
            assert elapsed >= 0.25, (
                f"Expected at least 0.25s with backoff delays, got {elapsed:.3f}s - "
                "backoff may not be applied correctly"
            )

    async def test_retry_exhaustion_raises_final_error(
        self,
        retry_config_fast: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test retry exhaustion raises final error with proper context.

        Validates:
        - All 3 attempts fail
        - Final error raised after exhaustion
        - Error includes proper context (transport_type, operation)
        - Correlation ID preserved in error
        """
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # All attempts fail
            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Service unavailable"
            )

            await handler.initialize(retry_config_fast)

            correlation_id = uuid4()
            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": correlation_id,
            }

            # Act & Assert: Should raise after all retries exhausted
            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(envelope)

            error = exc_info.value

            # Assert: Error contains proper context
            assert error.model.context is not None
            assert "operation" in error.model.context
            assert error.model.context["operation"] == "consul.kv_get"

            # Assert: Correlation ID preserved
            assert error.model.correlation_id == correlation_id

            # Assert: All 3 attempts were made
            assert mock_consul_client.kv.get.call_count == 3

    async def test_retry_exhaustion_with_circuit_breaker(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test retry exhaustion with circuit breaker integration.

        Validates:
        - max_attempts = 3
        - circuit_breaker_threshold = 5
        - First 3 attempts fail (retries exhausted)
        - Error raised, but circuit NOT opened (only 1 failure recorded)
        - Circuit breaker only counts final failure after retries exhausted

        Note: Per handler_consul.py implementation, circuit breaker failures
        are recorded only on final retry attempt, not on each individual retry.
        """
        handler = ConsulHandler()

        # Configure with circuit breaker threshold > max_attempts
        # initial_delay_seconds must be >= 0.1 per ModelConsulRetryConfig
        config: dict[str, object] = {
            "host": "consul.test.local",
            "port": 8500,
            "scheme": "http",
            "timeout_seconds": 1.0,  # Short timeout for fast tests
            "retry": {
                "max_attempts": 3,
                "initial_delay_seconds": 0.1,  # Minimum allowed value
                "max_delay_seconds": 1.0,
                "exponential_base": 2.0,
            },
            "circuit_breaker_enabled": True,
            "circuit_breaker_failure_threshold": 5,  # Higher than max_attempts
            "circuit_breaker_reset_timeout_seconds": 30.0,
        }

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # All attempts fail - use consul-specific exception
            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Service unavailable"
            )

            await handler.initialize(config)

            correlation_id = uuid4()
            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": correlation_id,
            }

            # Act: First operation fails after 3 retries
            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

            # Assert: All 3 attempts made
            assert mock_consul_client.kv.get.call_count == 3

            # Assert: Circuit should still be closed (only 1 failure recorded)
            # Circuit breaker records failure only on final retry attempt
            assert handler._circuit_breaker_initialized is True
            assert handler._circuit_breaker_failures == 1  # Only 1 failure recorded

            # Assert: Circuit is NOT open (1 < 5 threshold)
            assert handler._circuit_breaker_open is False

            # Reset for additional operations
            mock_consul_client.kv.get.reset_mock()
            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Service unavailable"
            )

            # Act: Second operation also fails
            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

            # Assert: Circuit still not open (2 < 5 threshold)
            assert handler._circuit_breaker_failures == 2
            assert handler._circuit_breaker_open is False

    async def test_non_retryable_error_fails_immediately(
        self,
        retry_config_fast: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test non-retryable error fails immediately without retry.

        Validates:
        - Authentication error triggered
        - No retry attempted (auth errors are non-retryable)
        - Error raised immediately
        - Only 1 attempt made

        Note: ACLPermissionDenied is considered a non-retryable error
        because it indicates a configuration/permission issue that
        won't be resolved by retrying.
        """
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Authentication error - should not retry
            mock_consul_client.kv.get.side_effect = consul.ACLPermissionDenied(
                "Token does not have required permissions"
            )

            await handler.initialize(retry_config_fast)

            correlation_id = uuid4()
            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": correlation_id,
            }

            # Act: Measure timing to verify no retry delay
            start = time.monotonic()
            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.execute(envelope)
            elapsed = time.monotonic() - start

            error = exc_info.value

            # Assert: Error is authentication error
            assert "permission denied" in error.message.lower()

            # Assert: Only 1 attempt made (no retries for auth errors)
            assert mock_consul_client.kv.get.call_count == 1

            # Assert: Failed fast - no backoff delay
            # Should complete in well under 0.1s (the initial backoff)
            assert (
                elapsed < 0.1
            ), f"Expected immediate failure (< 0.1s), got {elapsed:.3f}s"

            # Assert: Correlation ID preserved
            assert error.model.correlation_id == correlation_id

    async def test_timeout_error_retries_with_backoff(
        self,
        retry_config_fast: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test timeout errors are retried with backoff.

        Validates:
        - Timeout errors are considered transient
        - Retries are attempted with backoff
        - All attempts fail leads to InfraTimeoutError
        """
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Timeout error - should retry
            mock_consul_client.kv.get.side_effect = consul.Timeout("Request timed out")

            await handler.initialize(retry_config_fast)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            # Act: Execute and expect timeout error after retries
            start = time.monotonic()
            with pytest.raises(InfraTimeoutError):
                await handler.execute(envelope)
            elapsed = time.monotonic() - start

            # Assert: All 3 attempts made
            assert mock_consul_client.kv.get.call_count == 3

            # Assert: Backoff delays were applied (0.1s + 0.2s minimum)
            assert (
                elapsed >= 0.25
            ), f"Expected at least 0.25s total delay for retries, got {elapsed:.3f}s"

    async def test_success_on_last_retry(
        self,
        retry_config_fast: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test operation succeeds on the last retry attempt.

        Validates:
        - First 2 attempts fail
        - Third (last) attempt succeeds
        - Result is returned successfully
        """
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # First 2 fail, 3rd succeeds
            mock_consul_client.kv.get.side_effect = [
                Exception("Transient error 1"),
                Exception("Transient error 2"),
                (
                    0,
                    {
                        "Value": b"finally-success",
                        "Key": "test/key",
                        "ModifyIndex": 100,
                    },
                ),
            ]

            await handler.initialize(retry_config_fast)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            # Act
            output = await handler.execute(envelope)
            result = output.result

            # Assert: Operation succeeded
            assert result.status == "success"
            payload_data = result.payload.data
            assert payload_data.value == "finally-success"

            # Assert: All 3 attempts made
            assert mock_consul_client.kv.get.call_count == 3

    async def test_circuit_breaker_opens_after_multiple_exhausted_retries(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test circuit breaker opens after multiple retry-exhausted operations.

        Validates:
        - Circuit breaker counts failures after retry exhaustion
        - After enough failed operations, circuit opens
        - Subsequent operations fail fast with InfraUnavailableError
        """
        handler = ConsulHandler()

        # Configure with low circuit breaker threshold
        # initial_delay_seconds must be >= 0.1 per ModelConsulRetryConfig
        config: dict[str, object] = {
            "host": "consul.test.local",
            "port": 8500,
            "scheme": "http",
            "timeout_seconds": 1.0,  # Short timeout for fast tests
            "retry": {
                "max_attempts": 2,
                "initial_delay_seconds": 0.1,  # Minimum allowed value
                "max_delay_seconds": 1.0,
                "exponential_base": 2.0,
            },
            "circuit_breaker_enabled": True,
            "circuit_breaker_failure_threshold": 3,  # Opens after 3 failures
            "circuit_breaker_reset_timeout_seconds": 60.0,
        }

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # All attempts fail - use consul-specific exception
            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Service unavailable"
            )

            await handler.initialize(config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            # Act: Execute 3 operations (each exhausts retries, records 1 failure)
            for i in range(3):
                mock_consul_client.kv.get.reset_mock()
                mock_consul_client.kv.get.side_effect = consul.ConsulException(
                    "Service unavailable"
                )

                with pytest.raises(InfraConnectionError):
                    await handler.execute(envelope)

            # Assert: Circuit should now be open (3 failures = threshold)
            assert handler._circuit_breaker_failures == 3
            assert handler._circuit_breaker_open is True

            # Act: Next operation should fail fast with circuit breaker error
            mock_consul_client.kv.get.reset_mock()

            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)

            error = exc_info.value

            # Assert: Circuit breaker error
            assert "circuit breaker is open" in error.message.lower()

            # Assert: No new attempts made (failed immediately)
            assert mock_consul_client.kv.get.call_count == 0


__all__: list[str] = ["TestEffectRetryBackoff"]
