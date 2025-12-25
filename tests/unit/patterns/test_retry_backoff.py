# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for retry and exponential backoff patterns.

Tests cover:
- Retry policy respects max_attempts configuration
- Exponential backoff calculates delays correctly
- Jitter adds randomization within expected bounds (+/-25%)
- Max delay cap is respected
- Non-retryable errors fail immediately (no retry)
- Retryable errors trigger retry logic

These tests validate the retry patterns documented in CLAUDE.md and
implemented across handlers like VaultAdapter and ConsulHandler.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar
from uuid import UUID

import pytest

from omnibase_infra.handlers.model_consul_retry_config import ModelConsulRetryConfig
from omnibase_infra.handlers.model_vault_retry_config import ModelVaultRetryConfig

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Test Fixtures and Helper Classes
# ---------------------------------------------------------------------------


@dataclass
class RetryAttempt:
    """Records a single retry attempt for test verification."""

    attempt_number: int
    delay_before_seconds: float
    timestamp: float
    success: bool
    error: Exception | None = None


class MockRetryableOperation:
    """Simulates an operation that may fail and require retries.

    Tracks all attempts for verification of retry behavior.
    """

    def __init__(
        self,
        fail_count: int = 0,
        failure_exception: type[Exception] = Exception,
        success_result: str = "success",
    ) -> None:
        """Initialize mock operation.

        Args:
            fail_count: Number of times to fail before succeeding
            failure_exception: Exception type to raise on failure
            success_result: Result to return on success
        """
        self.fail_count = fail_count
        self.failure_exception = failure_exception
        self.success_result = success_result
        self.call_count = 0
        self.attempts: list[RetryAttempt] = []

    def __call__(self) -> str:
        """Execute the mock operation."""
        self.call_count += 1

        if self.call_count <= self.fail_count:
            error = self.failure_exception(f"Simulated failure {self.call_count}")
            self.attempts.append(
                RetryAttempt(
                    attempt_number=self.call_count,
                    delay_before_seconds=0.0,  # Set by caller
                    timestamp=time.time(),
                    success=False,
                    error=error,
                )
            )
            raise error

        self.attempts.append(
            RetryAttempt(
                attempt_number=self.call_count,
                delay_before_seconds=0.0,
                timestamp=time.time(),
                success=True,
            )
        )
        return self.success_result


class RetryExecutor:
    """Executes operations with retry and exponential backoff.

    This is a minimal implementation for testing the retry pattern.
    Production handlers (VaultAdapter, ConsulHandler) have more features.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        jitter_factor: float = 0.25,
        non_retryable_errors: frozenset[type[Exception]] | None = None,
    ) -> None:
        """Initialize retry executor with configuration.

        Args:
            max_attempts: Maximum retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay cap in seconds
            exponential_base: Exponential multiplier for backoff
            jitter_factor: Jitter randomization factor (+/- this percentage)
            non_retryable_errors: Exception types that should not trigger retry
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter_factor = jitter_factor
        self.non_retryable_errors = non_retryable_errors or frozenset()
        self.recorded_delays: list[float] = []

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Uses exponential backoff: initial_delay * (base ** attempt)
        Capped at max_delay.

        Args:
            attempt: Zero-indexed attempt number

        Returns:
            Delay in seconds before next attempt
        """
        delay = self.initial_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)

    def calculate_delay_with_jitter(self, attempt: int) -> float:
        """Calculate delay with jitter randomization.

        Adds +/- jitter_factor randomization to base delay.

        Args:
            attempt: Zero-indexed attempt number

        Returns:
            Delay with jitter applied
        """
        base_delay = self.calculate_delay(attempt)
        jitter_range = base_delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        return max(0.0, base_delay + jitter)

    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        correlation_id: UUID | None = None,
    ) -> T:
        """Execute operation with retry logic.

        Args:
            operation: Callable to execute
            correlation_id: Optional correlation ID for tracing

        Returns:
            Result from successful operation

        Raises:
            Exception: Last exception if all retries exhausted
            Non-retryable exceptions immediately without retry
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                return operation()
            except Exception as e:
                # Check if error is non-retryable
                if type(e) in self.non_retryable_errors:
                    raise

                last_exception = e

                # If not last attempt, calculate delay and sleep
                if attempt < self.max_attempts - 1:
                    delay = self.calculate_delay_with_jitter(attempt)
                    self.recorded_delays.append(delay)
                    await asyncio.sleep(delay)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry loop completed without result")


# ---------------------------------------------------------------------------
# Retry Config Model Tests
# ---------------------------------------------------------------------------


class TestVaultRetryConfig:
    """Test ModelVaultRetryConfig validation and defaults."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelVaultRetryConfig()
        assert config.max_attempts == 3
        assert config.initial_backoff_seconds == 0.1
        assert config.max_backoff_seconds == 10.0
        assert config.exponential_base == 2.0

    def test_custom_config(self) -> None:
        """Test custom configuration values are accepted."""
        config = ModelVaultRetryConfig(
            max_attempts=5,
            initial_backoff_seconds=0.5,
            max_backoff_seconds=30.0,
            exponential_base=3.0,
        )
        assert config.max_attempts == 5
        assert config.initial_backoff_seconds == 0.5
        assert config.max_backoff_seconds == 30.0
        assert config.exponential_base == 3.0

    def test_max_attempts_bounds(self) -> None:
        """Test max_attempts validation bounds (1-10)."""
        from pydantic import ValidationError

        # Valid minimum
        config = ModelVaultRetryConfig(max_attempts=1)
        assert config.max_attempts == 1

        # Valid maximum
        config = ModelVaultRetryConfig(max_attempts=10)
        assert config.max_attempts == 10

        # Below minimum
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultRetryConfig(max_attempts=0)
        assert "max_attempts" in str(exc_info.value)

        # Above maximum
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultRetryConfig(max_attempts=11)
        assert "max_attempts" in str(exc_info.value)

    def test_exponential_base_bounds(self) -> None:
        """Test exponential_base validation bounds (1.5-4.0)."""
        from pydantic import ValidationError

        # Valid minimum
        config = ModelVaultRetryConfig(exponential_base=1.5)
        assert config.exponential_base == 1.5

        # Valid maximum
        config = ModelVaultRetryConfig(exponential_base=4.0)
        assert config.exponential_base == 4.0

        # Below minimum
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultRetryConfig(exponential_base=1.0)
        assert "exponential_base" in str(exc_info.value)

        # Above maximum
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultRetryConfig(exponential_base=5.0)
        assert "exponential_base" in str(exc_info.value)

    def test_config_is_frozen(self) -> None:
        """Test config is immutable (frozen)."""
        config = ModelVaultRetryConfig()

        with pytest.raises(Exception):  # Pydantic raises on mutation
            config.max_attempts = 5  # type: ignore[misc]


class TestConsulRetryConfig:
    """Test ModelConsulRetryConfig validation and defaults."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelConsulRetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay_seconds == 1.0
        assert config.max_delay_seconds == 30.0
        assert config.exponential_base == 2.0

    def test_custom_config(self) -> None:
        """Test custom configuration values are accepted."""
        config = ModelConsulRetryConfig(
            max_attempts=5,
            initial_delay_seconds=2.0,
            max_delay_seconds=60.0,
            exponential_base=2.5,
        )
        assert config.max_attempts == 5
        assert config.initial_delay_seconds == 2.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 2.5


# ---------------------------------------------------------------------------
# Exponential Backoff Calculation Tests
# ---------------------------------------------------------------------------


class TestExponentialBackoffCalculation:
    """Test exponential backoff delay calculations."""

    def test_exponential_backoff_sequence(self) -> None:
        """Test backoff sequence follows exponential pattern.

        With initial=0.1, base=2.0:
        Attempt 0: 0.1 * 2^0 = 0.1s
        Attempt 1: 0.1 * 2^1 = 0.2s
        Attempt 2: 0.1 * 2^2 = 0.4s
        Attempt 3: 0.1 * 2^3 = 0.8s
        """
        executor = RetryExecutor(
            initial_delay=0.1,
            max_delay=10.0,
            exponential_base=2.0,
        )

        assert executor.calculate_delay(0) == pytest.approx(0.1, rel=1e-6)
        assert executor.calculate_delay(1) == pytest.approx(0.2, rel=1e-6)
        assert executor.calculate_delay(2) == pytest.approx(0.4, rel=1e-6)
        assert executor.calculate_delay(3) == pytest.approx(0.8, rel=1e-6)

    def test_exponential_backoff_with_base_3(self) -> None:
        """Test backoff with base=3.0.

        With initial=1.0, base=3.0:
        Attempt 0: 1.0 * 3^0 = 1.0s
        Attempt 1: 1.0 * 3^1 = 3.0s
        Attempt 2: 1.0 * 3^2 = 9.0s
        """
        executor = RetryExecutor(
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=3.0,
        )

        assert executor.calculate_delay(0) == pytest.approx(1.0, rel=1e-6)
        assert executor.calculate_delay(1) == pytest.approx(3.0, rel=1e-6)
        assert executor.calculate_delay(2) == pytest.approx(9.0, rel=1e-6)

    def test_max_delay_cap_is_respected(self) -> None:
        """Test that max_delay caps the backoff.

        With initial=1.0, base=2.0, max=5.0:
        Attempt 3: 1.0 * 2^3 = 8.0 -> capped to 5.0
        Attempt 4: 1.0 * 2^4 = 16.0 -> capped to 5.0
        """
        executor = RetryExecutor(
            initial_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
        )

        # Below cap
        assert executor.calculate_delay(0) == pytest.approx(1.0, rel=1e-6)
        assert executor.calculate_delay(1) == pytest.approx(2.0, rel=1e-6)
        assert executor.calculate_delay(2) == pytest.approx(4.0, rel=1e-6)

        # At/above cap
        assert executor.calculate_delay(3) == pytest.approx(5.0, rel=1e-6)  # 8.0 -> 5.0
        assert executor.calculate_delay(4) == pytest.approx(
            5.0, rel=1e-6
        )  # 16.0 -> 5.0
        assert executor.calculate_delay(10) == pytest.approx(5.0, rel=1e-6)


class TestJitterRandomization:
    """Test jitter adds randomization within expected bounds."""

    def test_jitter_within_bounds(self) -> None:
        """Test jitter stays within +/- 25% of base delay.

        With base delay = 1.0s and jitter_factor = 0.25:
        - Minimum: 1.0 - 0.25 = 0.75s
        - Maximum: 1.0 + 0.25 = 1.25s
        """
        executor = RetryExecutor(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter_factor=0.25,
        )

        # Run many iterations to verify bounds
        samples = 1000
        min_delay = float("inf")
        max_delay = float("-inf")

        for _ in range(samples):
            delay = executor.calculate_delay_with_jitter(0)  # Base delay = 1.0
            min_delay = min(min_delay, delay)
            max_delay = max(max_delay, delay)

        # Should be within +/- 25% of 1.0
        assert min_delay >= 0.75 - 0.01  # Small tolerance
        assert max_delay <= 1.25 + 0.01

    def test_jitter_provides_variance(self) -> None:
        """Test jitter provides actual variance (not constant)."""
        executor = RetryExecutor(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter_factor=0.25,
        )

        # Collect samples
        samples = [executor.calculate_delay_with_jitter(0) for _ in range(100)]

        # Calculate variance
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)

        # Variance should be non-trivial (jitter is working)
        assert variance > 0.001, "Jitter should add meaningful variance"

    def test_jitter_with_different_factors(self) -> None:
        """Test different jitter factors produce different bounds."""
        # 10% jitter
        executor_10 = RetryExecutor(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter_factor=0.10,
        )

        # 50% jitter
        executor_50 = RetryExecutor(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter_factor=0.50,
        )

        samples_10 = [executor_10.calculate_delay_with_jitter(0) for _ in range(100)]
        samples_50 = [executor_50.calculate_delay_with_jitter(0) for _ in range(100)]

        # 10% jitter should have smaller range
        range_10 = max(samples_10) - min(samples_10)
        range_50 = max(samples_50) - min(samples_50)

        assert range_50 > range_10, "Higher jitter factor should produce larger range"


# ---------------------------------------------------------------------------
# Retry Logic Tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Test retry logic respects max_attempts configuration."""

    @pytest.mark.asyncio
    async def test_respects_max_attempts(self) -> None:
        """Test operation is attempted exactly max_attempts times on persistent failure."""
        operation = MockRetryableOperation(fail_count=100)  # Always fail
        executor = RetryExecutor(
            max_attempts=3,
            initial_delay=0.001,  # Fast for tests
            max_delay=0.01,
        )

        with pytest.raises(Exception) as exc_info:
            await executor.execute_with_retry(operation)

        assert "Simulated failure" in str(exc_info.value)
        assert operation.call_count == 3  # Exactly max_attempts

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        """Test no retry when operation succeeds immediately."""
        operation = MockRetryableOperation(fail_count=0)  # Success on first try
        executor = RetryExecutor(
            max_attempts=3,
            initial_delay=0.001,
        )

        result = await executor.execute_with_retry(operation)

        assert result == "success"
        assert operation.call_count == 1  # Only one attempt needed

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_failures(self) -> None:
        """Test retry succeeds after transient failures."""
        operation = MockRetryableOperation(fail_count=2)  # Fail twice, then succeed
        executor = RetryExecutor(
            max_attempts=5,
            initial_delay=0.001,
        )

        result = await executor.execute_with_retry(operation)

        assert result == "success"
        assert operation.call_count == 3  # Two failures + one success

    @pytest.mark.asyncio
    async def test_records_delays_between_retries(self) -> None:
        """Test delays are recorded between retry attempts."""
        operation = MockRetryableOperation(fail_count=3)  # Fail three times
        executor = RetryExecutor(
            max_attempts=5,
            initial_delay=0.01,
            exponential_base=2.0,
            jitter_factor=0.0,  # No jitter for deterministic test
        )

        await executor.execute_with_retry(operation)

        # Should have 3 delays recorded (for attempts 0, 1, 2)
        assert len(executor.recorded_delays) == 3
        assert executor.recorded_delays[0] == pytest.approx(0.01, rel=0.01)  # 0.01
        assert executor.recorded_delays[1] == pytest.approx(0.02, rel=0.01)  # 0.02
        assert executor.recorded_delays[2] == pytest.approx(0.04, rel=0.01)  # 0.04


class TestNonRetryableErrors:
    """Test non-retryable errors fail immediately without retry."""

    @pytest.mark.asyncio
    async def test_non_retryable_error_no_retry(self) -> None:
        """Test non-retryable errors fail immediately."""

        class AuthenticationError(Exception):
            """Non-retryable authentication error."""

        operation = MockRetryableOperation(
            fail_count=100,
            failure_exception=AuthenticationError,
        )
        executor = RetryExecutor(
            max_attempts=5,
            initial_delay=0.001,
            non_retryable_errors=frozenset({AuthenticationError}),
        )

        with pytest.raises(AuthenticationError):
            await executor.execute_with_retry(operation)

        # Should fail on first attempt, no retries
        assert operation.call_count == 1

    @pytest.mark.asyncio
    async def test_retryable_error_retries(self) -> None:
        """Test retryable errors trigger retry logic."""

        class TransientError(Exception):
            """Retryable transient error."""

        class AuthenticationError(Exception):
            """Non-retryable authentication error."""

        operation = MockRetryableOperation(
            fail_count=100,
            failure_exception=TransientError,
        )
        executor = RetryExecutor(
            max_attempts=3,
            initial_delay=0.001,
            non_retryable_errors=frozenset(
                {AuthenticationError}
            ),  # TransientError not here
        )

        with pytest.raises(TransientError):
            await executor.execute_with_retry(operation)

        # Should retry all attempts
        assert operation.call_count == 3

    @pytest.mark.asyncio
    async def test_mixed_error_types(self) -> None:
        """Test first non-retryable error stops retries even after retryable ones."""

        class TransientError(Exception):
            pass

        class FatalError(Exception):
            pass

        call_count = 0
        errors = [TransientError(), TransientError(), FatalError()]

        def operation_with_mixed_errors() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= len(errors):
                raise errors[call_count - 1]
            return "success"

        executor = RetryExecutor(
            max_attempts=5,
            initial_delay=0.001,
            non_retryable_errors=frozenset({FatalError}),
        )

        with pytest.raises(FatalError):
            await executor.execute_with_retry(operation_with_mixed_errors)

        # Should stop at third call (FatalError)
        assert call_count == 3


# ---------------------------------------------------------------------------
# Integration-Style Tests
# ---------------------------------------------------------------------------


class TestRetryWithRealConfig:
    """Integration tests using actual config models."""

    @pytest.mark.asyncio
    async def test_vault_retry_config_integration(self) -> None:
        """Test retry behavior with ModelVaultRetryConfig values.

        Uses minimum valid bounds from the config model:
        - initial_backoff_seconds: >= 0.01
        - max_backoff_seconds: >= 1.0
        """
        config = ModelVaultRetryConfig(
            max_attempts=3,
            initial_backoff_seconds=0.1,  # Minimum is 0.01
            max_backoff_seconds=1.0,  # Minimum is 1.0
            exponential_base=2.0,
        )

        executor = RetryExecutor(
            max_attempts=config.max_attempts,
            initial_delay=config.initial_backoff_seconds,
            max_delay=config.max_backoff_seconds,
            exponential_base=config.exponential_base,
            jitter_factor=0.0,  # Disable jitter for deterministic test
        )

        # Verify the config is correctly applied to executor
        assert executor.max_attempts == 3
        assert executor.initial_delay == 0.1
        assert executor.max_delay == 1.0
        assert executor.exponential_base == 2.0

        # Verify exponential backoff calculation with config values
        # Attempt 0: 0.1 * 2^0 = 0.1
        # Attempt 1: 0.1 * 2^1 = 0.2
        # Attempt 2: 0.1 * 2^2 = 0.4
        assert executor.calculate_delay(0) == pytest.approx(0.1, rel=0.01)
        assert executor.calculate_delay(1) == pytest.approx(0.2, rel=0.01)
        assert executor.calculate_delay(2) == pytest.approx(0.4, rel=0.01)

    @pytest.mark.asyncio
    async def test_consul_retry_config_integration(self) -> None:
        """Test retry behavior with ModelConsulRetryConfig values.

        Uses minimum valid bounds from the config model:
        - initial_delay_seconds: >= 0.1
        - max_delay_seconds: >= 1.0
        """
        config = ModelConsulRetryConfig(
            max_attempts=4,
            initial_delay_seconds=0.5,  # Minimum is 0.1
            max_delay_seconds=2.0,  # Minimum is 1.0
            exponential_base=2.0,
        )

        executor = RetryExecutor(
            max_attempts=config.max_attempts,
            initial_delay=config.initial_delay_seconds,
            max_delay=config.max_delay_seconds,
            exponential_base=config.exponential_base,
            jitter_factor=0.0,
        )

        # Verify the config is correctly applied to executor
        assert executor.max_attempts == 4
        assert executor.initial_delay == 0.5
        assert executor.max_delay == 2.0

        # Verify max_delay cap is respected
        # Attempt 0: 0.5 * 2^0 = 0.5 (under cap)
        # Attempt 1: 0.5 * 2^1 = 1.0 (under cap)
        # Attempt 2: 0.5 * 2^2 = 2.0 (at cap)
        # Attempt 3: 0.5 * 2^3 = 4.0 -> capped to 2.0
        assert executor.calculate_delay(0) == pytest.approx(0.5, rel=0.01)
        assert executor.calculate_delay(1) == pytest.approx(1.0, rel=0.01)
        assert executor.calculate_delay(2) == pytest.approx(2.0, rel=0.01)
        assert executor.calculate_delay(3) == pytest.approx(2.0, rel=0.01)  # Capped


__all__: list[str] = [
    "TestVaultRetryConfig",
    "TestConsulRetryConfig",
    "TestExponentialBackoffCalculation",
    "TestJitterRandomization",
    "TestRetryLogic",
    "TestNonRetryableErrors",
    "TestRetryWithRealConfig",
]
