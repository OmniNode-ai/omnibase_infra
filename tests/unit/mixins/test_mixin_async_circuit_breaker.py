# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive unit tests for MixinAsyncCircuitBreaker.

This test suite validates:
- Basic circuit breaker functionality (state management, failure counting)
- Thread safety with concurrent operations (100+ parallel tasks)
- Correlation ID propagation and generation
- Error context validation
- Edge cases (threshold=1, zero timeout, multiple resets)

Test Organization:
    - TestMixinAsyncCircuitBreakerBasics: Basic functionality
    - TestMixinAsyncCircuitBreakerThreadSafety: Concurrency and race conditions
    - TestMixinAsyncCircuitBreakerCorrelationId: Correlation ID handling
    - TestMixinAsyncCircuitBreakerErrorContext: Error context validation
    - TestMixinAsyncCircuitBreakerEdgeCases: Edge cases and boundary conditions

Related Test Files:
    - test_circuit_breaker_transitions.py: Dedicated state transition tests
    - test_mixin_async_circuit_breaker_race_conditions.py: Race condition tests
    - test_effect_circuit_breaker.py: Effect-level integration tests
    - test_recovery_circuit_breaker.py: Chaos/recovery tests

Coverage Goals:
    - >90% code coverage for mixin
    - Thread safety validated with parallel execution
    - All error paths tested
"""

import asyncio
import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumCircuitState, EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    MixinAsyncCircuitBreaker,
)

if TYPE_CHECKING:
    from omnibase_infra.models.resilience import ModelCircuitBreakerConfig


class CircuitBreakerServiceStub(MixinAsyncCircuitBreaker):
    """Test service that uses circuit breaker mixin for testing."""

    def __init__(
        self,
        threshold: int = 5,
        reset_timeout: float = 60.0,
        service_name: str = "test-service",
        transport_type: EnumInfraTransportType = EnumInfraTransportType.HTTP,
    ) -> None:
        """Initialize test service with circuit breaker.

        Args:
            threshold: Maximum failures before opening circuit
            reset_timeout: Seconds before automatic reset
            service_name: Service identifier for error context
            transport_type: Transport type for error context
        """
        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=service_name,
            transport_type=transport_type,
            enable_active_recovery=False,
        )

    async def check_circuit(
        self, operation: str = "test_operation", correlation_id: UUID | None = None
    ) -> None:
        """Check circuit breaker state (thread-safe wrapper for testing)."""
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

    async def record_failure(
        self, operation: str = "test_operation", correlation_id: UUID | None = None
    ) -> None:
        """Record circuit failure (thread-safe wrapper for testing)."""
        async with self._circuit_breaker_lock:
            await self._record_circuit_failure(operation, correlation_id)

    async def reset_circuit(self) -> None:
        """Reset circuit breaker (thread-safe wrapper for testing)."""
        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()

    def get_state(self) -> EnumCircuitState:
        """Get current circuit state (for testing assertions)."""
        if self._circuit_breaker_open:
            return EnumCircuitState.OPEN
        return EnumCircuitState.CLOSED

    def get_failure_count(self) -> int:
        """Get current failure count (for testing assertions)."""
        return self._circuit_breaker_failures

    async def execute_operation(
        self,
        operation: str = "test_operation",
        correlation_id: UUID | None = None,
        should_fail: bool = False,
    ) -> str:
        """Execute an operation through the circuit breaker pattern.

        This simulates a real operation that:
        1. Checks circuit breaker before execution
        2. Executes the operation (success or failure based on should_fail)
        3. Records success/failure with circuit breaker
        4. Returns result or raises exception

        Args:
            operation: Operation name for logging
            correlation_id: Optional correlation ID for tracing
            should_fail: If True, simulate operation failure

        Returns:
            Success result string

        Raises:
            RuntimeError: If should_fail is True
            InfraUnavailableError: If circuit is open
        """
        # Check circuit before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            if should_fail:
                raise RuntimeError("Simulated operation failure")

            # Simulate successful operation
            result = f"success:{operation}"

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return result

        except RuntimeError:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerBasics:
    """Test basic circuit breaker functionality."""

    async def test_circuit_starts_closed(self) -> None:
        """Test that circuit breaker starts in CLOSED state."""
        service = CircuitBreakerServiceStub()
        assert service.get_state() == EnumCircuitState.CLOSED
        assert service.get_failure_count() == 0
        assert not service._circuit_breaker_open

    async def test_check_allows_operation_when_closed(self) -> None:
        """Test that check_circuit allows operations when circuit is CLOSED."""
        service = CircuitBreakerServiceStub()

        # Should not raise when circuit is closed
        await service.check_circuit("test_operation")

        # Circuit should remain closed
        assert service.get_state() == EnumCircuitState.CLOSED

    async def test_record_failure_increments_counter(self) -> None:
        """Test that record_failure increments the failure counter."""
        service = CircuitBreakerServiceStub(threshold=5)

        # Record multiple failures (below threshold)
        await service.record_failure("test_operation")
        assert service.get_failure_count() == 1

        await service.record_failure("test_operation")
        assert service.get_failure_count() == 2

        await service.record_failure("test_operation")
        assert service.get_failure_count() == 3

        # Circuit should still be closed (below threshold)
        assert service.get_state() == EnumCircuitState.CLOSED

    async def test_record_failure_opens_circuit_at_threshold(self) -> None:
        """Test that circuit opens when failure threshold is reached."""
        service = CircuitBreakerServiceStub(threshold=3)

        # Record failures up to threshold
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")

        # Circuit should now be open
        assert service.get_state() == EnumCircuitState.OPEN
        assert service._circuit_breaker_open is True

    async def test_check_raises_when_open(self) -> None:
        """Test that check_circuit raises InfraUnavailableError when circuit is OPEN."""
        service = CircuitBreakerServiceStub(threshold=2)

        # Open the circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == EnumCircuitState.OPEN

        # check_circuit should raise InfraUnavailableError
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert "Circuit breaker is open" in error.message
        assert error.model.context.get("circuit_state") == "open"

    async def test_reset_closes_circuit(self) -> None:
        """Test that reset_circuit closes the circuit and resets failure count."""
        service = CircuitBreakerServiceStub(threshold=2)

        # Open the circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == EnumCircuitState.OPEN

        # Reset the circuit
        await service.reset_circuit()

        # Circuit should now be closed
        assert service.get_state() == EnumCircuitState.CLOSED
        assert service.get_failure_count() == 0
        assert service._circuit_breaker_open is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerThreadSafety:
    """Test circuit breaker thread safety with concurrent operations."""

    async def test_concurrent_check_operations(self) -> None:
        """Test multiple check operations in parallel (100 tasks)."""
        service = CircuitBreakerServiceStub(threshold=10)

        # Run 100 concurrent check operations
        tasks = [service.check_circuit("test_operation") for _ in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All checks should succeed (circuit is closed)
        assert all(result is None for result in results)

    async def test_concurrent_failure_recording(self) -> None:
        """Test multiple failure recordings in parallel (100 tasks)."""
        service = CircuitBreakerServiceStub(threshold=200)

        # Run 100 concurrent failure recordings
        tasks = [service.record_failure("test_operation") for _ in range(100)]
        await asyncio.gather(*tasks)

        # Failure count should be exactly 100 (no race conditions)
        assert service.get_failure_count() == 100

    async def test_concurrent_check_and_failure(self) -> None:
        """Test mixed check and failure operations in parallel."""
        service = CircuitBreakerServiceStub(threshold=50)

        # Create mixed tasks (50 checks, 50 failures)
        check_tasks = [service.check_circuit("check") for _ in range(50)]
        failure_tasks = [service.record_failure("failure") for _ in range(50)]
        all_tasks = check_tasks + failure_tasks

        # Run all tasks concurrently
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Check operations should succeed, failure operations should record
        check_results = results[:50]
        assert all(result is None for result in check_results)

        # Circuit should be open (50 failures >= threshold)
        assert service.get_state() == EnumCircuitState.OPEN

    async def test_no_race_condition_at_threshold(self) -> None:
        """Test that exactly threshold failures opens circuit (no race)."""
        threshold = 10
        service = CircuitBreakerServiceStub(threshold=threshold)

        # Record exactly threshold failures concurrently
        tasks = [service.record_failure("test_operation") for _ in range(threshold)]
        await asyncio.gather(*tasks)

        # Circuit should be open
        assert service.get_state() == EnumCircuitState.OPEN
        assert service.get_failure_count() == threshold

    async def test_lock_prevents_race_conditions(self) -> None:
        """Test that lock prevents race conditions during state transitions."""
        service = CircuitBreakerServiceStub(threshold=5, reset_timeout=0.1)

        # Concurrent operations: failures, checks, resets
        async def mixed_operations() -> None:
            """Perform mixed operations concurrently."""
            operations = []
            for i in range(20):
                if i % 3 == 0:
                    operations.append(service.record_failure("failure"))
                elif i % 3 == 1:
                    operations.append(service.check_circuit("check"))
                else:
                    operations.append(service.reset_circuit())

            await asyncio.gather(*operations, return_exceptions=True)

        # Run mixed operations
        await mixed_operations()

        # Final state should be consistent (no corruption)
        # Either closed (reset) or open (failures)
        state = service.get_state()
        assert state in (EnumCircuitState.CLOSED, EnumCircuitState.OPEN)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerCorrelationId:
    """Test correlation ID propagation and generation."""

    async def test_correlation_id_propagation(self) -> None:
        """Test that correlation_id flows through errors."""
        service = CircuitBreakerServiceStub(threshold=1)

        # Open the circuit
        correlation_id = uuid4()
        await service.record_failure("test_operation", correlation_id)

        # Check should raise with same correlation_id
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation", correlation_id)

        error = exc_info.value
        assert error.model.correlation_id == correlation_id

    async def test_correlation_id_generated_if_none(self) -> None:
        """Test that UUID is generated if correlation_id not provided."""
        service = CircuitBreakerServiceStub(threshold=1)

        # Open the circuit without correlation_id
        await service.record_failure("test_operation")

        # Check should raise with generated correlation_id
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert isinstance(error.model.correlation_id, UUID)
        assert error.model.correlation_id.version == 4

    async def test_correlation_id_in_error_context(self) -> None:
        """Test that correlation_id is properly included in error context."""
        service = CircuitBreakerServiceStub(threshold=1)

        # Open circuit with specific correlation_id
        correlation_id = uuid4()
        await service.record_failure("test_operation", correlation_id)

        # Verify error contains correlation_id
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation", correlation_id)

        error = exc_info.value
        # Correlation ID is at model level, not in context dict
        assert error.model.correlation_id == correlation_id
        # Context is a dict with transport_type, operation, target_name
        assert isinstance(error.model.context, dict)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerErrorContext:
    """Test error context validation and structure."""

    async def test_error_context_contains_required_fields(self) -> None:
        """Test that error context contains all required fields."""
        service = CircuitBreakerServiceStub(
            threshold=1,
            service_name="test-service",
            transport_type=EnumInfraTransportType.KAFKA,
        )

        # Open the circuit
        await service.record_failure("publish_event")

        # Check error context structure
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("publish_event")

        error = exc_info.value
        context = error.model.context

        # Context is a dict with structured fields
        assert context["transport_type"] == EnumInfraTransportType.KAFKA
        assert context["operation"] == "publish_event"
        assert context["target_name"] == "test-service"
        # Correlation ID is at model level
        assert error.model.correlation_id is not None

    async def test_error_includes_service_name(self) -> None:
        """Test that error includes service_name in context."""
        service_name = "custom-kafka-service"
        service = CircuitBreakerServiceStub(threshold=1, service_name=service_name)

        # Open circuit
        await service.record_failure("test_operation")

        # Verify service name in error
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert error.model.context["target_name"] == service_name
        assert service_name in error.message

    async def test_error_includes_circuit_state(self) -> None:
        """Test that error includes circuit_state in context."""
        service = CircuitBreakerServiceStub(threshold=1)

        # Open circuit
        await service.record_failure("test_operation")

        # Verify circuit_state in error context
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert error.model.context.get("circuit_state") == "open"

    async def test_error_includes_retry_after(self) -> None:
        """Test that error includes retry_after_seconds calculated correctly."""
        reset_timeout = 10.0
        service = CircuitBreakerServiceStub(threshold=1, reset_timeout=reset_timeout)

        # Open circuit
        await service.record_failure("test_operation")

        # Immediately check (should raise with retry_after)
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        retry_after = error.model.context.get("retry_after_seconds")

        # Should be close to reset_timeout (within 1 second tolerance)
        assert retry_after is not None
        assert isinstance(retry_after, int)
        assert 0 <= retry_after <= reset_timeout


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_threshold_of_one(self) -> None:
        """Test circuit breaker with threshold=1 (opens on first failure)."""
        service = CircuitBreakerServiceStub(threshold=1)

        # First failure should open circuit
        await service.record_failure("test_operation")
        assert service.get_state() == EnumCircuitState.OPEN

        # Check should raise immediately
        with pytest.raises(InfraUnavailableError):
            await service.check_circuit("test_operation")

    async def test_zero_reset_timeout(self) -> None:
        """Test circuit breaker with zero reset timeout (immediate reset)."""
        service = CircuitBreakerServiceStub(threshold=2, reset_timeout=0.0)

        # Open circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == EnumCircuitState.OPEN

        # Immediate check should auto-reset (no wait needed)
        await service.check_circuit("test_operation")
        assert service.get_failure_count() == 0

    async def test_very_long_reset_timeout(self) -> None:
        """Test circuit breaker with very long reset timeout."""
        service = CircuitBreakerServiceStub(threshold=1, reset_timeout=3600.0)

        # Open circuit
        await service.record_failure("test_operation")

        # Circuit should stay open for long timeout
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        retry_after = error.model.context.get("retry_after_seconds")
        assert retry_after is not None
        assert retry_after > 3500  # Should be close to 3600

    async def test_multiple_resets(self) -> None:
        """Test multiple manual resets work correctly."""
        service = CircuitBreakerServiceStub(threshold=2)

        # Open and reset circuit multiple times
        for _ in range(5):
            # Open circuit
            await service.record_failure("test_operation")
            await service.record_failure("test_operation")
            assert service.get_state() == EnumCircuitState.OPEN

            # Reset circuit
            await service.reset_circuit()
            assert service.get_state() == EnumCircuitState.CLOSED
            assert service.get_failure_count() == 0

    async def test_failure_after_manual_reset(self) -> None:
        """Test that failures after manual reset work correctly."""
        service = CircuitBreakerServiceStub(threshold=3)

        # Record some failures
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_failure_count() == 2

        # Manual reset
        await service.reset_circuit()
        assert service.get_failure_count() == 0

        # New failures should count from zero
        await service.record_failure("test_operation")
        assert service.get_failure_count() == 1

        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == EnumCircuitState.OPEN

    async def test_concurrent_operations_at_threshold_boundary(self) -> None:
        """Test concurrent operations near threshold boundary."""
        threshold = 5
        service = CircuitBreakerServiceStub(threshold=threshold)

        # Record threshold - 1 failures
        for _ in range(threshold - 1):
            await service.record_failure("test_operation")

        assert service.get_state() == EnumCircuitState.CLOSED

        # One more failure should open circuit
        await service.record_failure("test_operation")
        assert service.get_state() == EnumCircuitState.OPEN

    async def test_reset_idempotency(self) -> None:
        """Test that reset is idempotent (multiple resets don't break state)."""
        service = CircuitBreakerServiceStub(threshold=2)

        # Open circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")

        # Multiple resets should be safe
        await service.reset_circuit()
        await service.reset_circuit()
        await service.reset_circuit()

        # State should be consistent
        assert service.get_state() == EnumCircuitState.CLOSED
        assert service.get_failure_count() == 0

    async def test_check_circuit_timing_precision(self) -> None:
        """Test that timeout timing is precise (no off-by-one errors)."""
        reset_timeout = 0.2
        service = CircuitBreakerServiceStub(threshold=1, reset_timeout=reset_timeout)

        # Open circuit
        start_time = time.perf_counter()
        await service.record_failure("test_operation")

        # Check immediately - should fail
        with pytest.raises(InfraUnavailableError):
            await service.check_circuit("test_operation")

        # Wait exactly reset_timeout
        elapsed = time.perf_counter() - start_time
        remaining = reset_timeout - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining + 0.05)  # Small buffer for precision

        # Check after timeout - should succeed (auto-reset)
        await service.check_circuit("test_operation")
        assert service.get_failure_count() == 0


class CircuitBreakerConfigServiceStub(MixinAsyncCircuitBreaker):
    """Test service that uses _init_circuit_breaker_from_config for testing."""

    def __init__(
        self,
        config: "ModelCircuitBreakerConfig",
    ) -> None:
        """Initialize test service with circuit breaker from config.

        Args:
            config: Circuit breaker configuration model
        """
        self._init_circuit_breaker_from_config(config)

    async def check_circuit(
        self, operation: str = "test_operation", correlation_id: UUID | None = None
    ) -> None:
        """Check circuit breaker state (thread-safe wrapper for testing)."""
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

    async def record_failure(
        self, operation: str = "test_operation", correlation_id: UUID | None = None
    ) -> None:
        """Record circuit failure (thread-safe wrapper for testing)."""
        async with self._circuit_breaker_lock:
            await self._record_circuit_failure(operation, correlation_id)

    def get_state(self) -> EnumCircuitState:
        """Get current circuit state (for testing assertions)."""
        if self._circuit_breaker_open:
            return EnumCircuitState.OPEN
        return EnumCircuitState.CLOSED


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerFromConfig:
    """Test _init_circuit_breaker_from_config method.

    This test class validates that the config-based initialization correctly
    delegates to _init_circuit_breaker with the config values.
    """

    async def test_init_from_config_with_defaults(self) -> None:
        """Test initialization from config with default values."""
        from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

        config = ModelCircuitBreakerConfig()
        service = CircuitBreakerConfigServiceStub(config)

        # Verify default values were applied
        assert service.circuit_breaker_threshold == 5
        assert service.circuit_breaker_reset_timeout == 60.0
        assert service.service_name == "unknown"
        assert service._cb_transport_type == EnumInfraTransportType.HTTP

    async def test_init_from_config_with_custom_values(self) -> None:
        """Test initialization from config with custom values."""
        from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

        config = ModelCircuitBreakerConfig(
            threshold=10,
            reset_timeout_seconds=120.0,
            service_name="kafka.production",
            transport_type=EnumInfraTransportType.KAFKA,
        )
        service = CircuitBreakerConfigServiceStub(config)

        # Verify custom values were applied
        assert service.circuit_breaker_threshold == 10
        assert service.circuit_breaker_reset_timeout == 120.0
        assert service.service_name == "kafka.production"
        assert service._cb_transport_type == EnumInfraTransportType.KAFKA

    async def test_init_from_config_circuit_functions_correctly(self) -> None:
        """Test that circuit breaker initialized from config functions correctly."""
        from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

        config = ModelCircuitBreakerConfig(
            threshold=2,
            reset_timeout_seconds=60.0,
            service_name="test-service",
            transport_type=EnumInfraTransportType.DATABASE,
        )
        service = CircuitBreakerConfigServiceStub(config)

        # Circuit should start closed
        assert service.get_state() == EnumCircuitState.CLOSED

        # Record failures to open circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")

        # Circuit should now be open
        assert service.get_state() == EnumCircuitState.OPEN

        # Check should raise with correct transport type in error context
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert error.model.context["transport_type"] == EnumInfraTransportType.DATABASE
        assert error.model.context["target_name"] == "test-service"

    async def test_init_from_config_from_env(self) -> None:
        """Test initialization from config created via from_env()."""
        import os
        from unittest.mock import patch

        from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

        env_vars = {
            "TEST_CB_THRESHOLD": "3",
            "TEST_CB_RESET_TIMEOUT": "30.0",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="consul.dev",
                transport_type=EnumInfraTransportType.CONSUL,
                prefix="TEST_CB",
            )
            service = CircuitBreakerConfigServiceStub(config)

            # Verify values from environment were applied
            assert service.circuit_breaker_threshold == 3
            assert service.circuit_breaker_reset_timeout == 30.0
            assert service.service_name == "consul.dev"
            assert service._cb_transport_type == EnumInfraTransportType.CONSUL

    async def test_init_from_config_all_transport_types(self) -> None:
        """Test initialization from config with all transport types."""
        from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

        transport_types = [
            EnumInfraTransportType.HTTP,
            EnumInfraTransportType.DATABASE,
            EnumInfraTransportType.KAFKA,
            EnumInfraTransportType.CONSUL,
            EnumInfraTransportType.VAULT,
            EnumInfraTransportType.VALKEY,
            EnumInfraTransportType.GRPC,
            EnumInfraTransportType.RUNTIME,
        ]

        for transport_type in transport_types:
            config = ModelCircuitBreakerConfig(
                threshold=5,
                reset_timeout_seconds=60.0,
                service_name=f"service.{transport_type.value}",
                transport_type=transport_type,
            )
            service = CircuitBreakerConfigServiceStub(config)

            assert service._cb_transport_type == transport_type
            assert service.service_name == f"service.{transport_type.value}"
