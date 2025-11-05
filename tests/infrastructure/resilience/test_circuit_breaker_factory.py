"""
Comprehensive tests for Circuit Breaker Factory.

Tests circuit breaker creation, state transitions, error handling,
and OnexError integration.
"""

import pytest
from circuitbreaker import CircuitBreakerError
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from omnibase_infra.infrastructure.resilience.circuit_breaker_factory import (
    InfrastructureCircuitBreaker,
    create_database_circuit_breaker,
    create_kafka_circuit_breaker,
    create_network_circuit_breaker,
    create_vault_circuit_breaker,
)


class TestInfrastructureCircuitBreakerInit:
    """Test circuit breaker initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        cb = InfrastructureCircuitBreaker()

        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.expected_exception == Exception

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        cb = InfrastructureCircuitBreaker(
            failure_threshold=10,
            recovery_timeout=30,
            expected_exception=ValueError,
            name="test_breaker",
        )

        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 30
        assert cb.expected_exception == ValueError
        assert cb.name == "test_breaker"

    def test_init_with_fallback(self):
        """Test initialization with fallback function."""

        def fallback_func():
            return "fallback_value"

        cb = InfrastructureCircuitBreaker(
            fallback_function=fallback_func,
        )

        assert cb.fallback_function == fallback_func


class TestInfrastructureCircuitBreakerExecution:
    """Test circuit breaker function execution."""

    def test_successful_execution(self):
        """Test successful function execution through circuit breaker."""
        cb = InfrastructureCircuitBreaker(failure_threshold=3)

        @cb
        def successful_func():
            return "success"

        result = successful_func()

        assert result == "success"
        assert cb.failure_count == 0

    def test_failed_execution_under_threshold(self):
        """Test failed execution under failure threshold."""
        cb = InfrastructureCircuitBreaker(failure_threshold=3)

        @cb
        def failing_func():
            raise ValueError("Test error")

        # Fail twice (under threshold)
        for _ in range(2):
            with pytest.raises(ValueError):
                failing_func()

        assert cb.failure_count == 2
        assert cb.state == "closed"

    def test_circuit_opens_at_threshold(self):
        """Test circuit opens at failure threshold."""
        cb = InfrastructureCircuitBreaker(failure_threshold=3)

        @cb
        def failing_func():
            raise ValueError("Test error")

        # Fail 3 times to reach threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                failing_func()

        assert cb.state == "open"

    def test_circuit_open_raises_onex_error(self):
        """Test that open circuit raises OnexError."""
        cb = InfrastructureCircuitBreaker(
            failure_threshold=2,
            name="test_service",
        )

        @cb
        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                failing_func()

        # Next call should raise OnexError
        with pytest.raises(OnexError) as exc_info:
            failing_func()

        assert exc_info.value.code == CoreErrorCode.SERVICE_UNAVAILABLE
        assert "Circuit breaker open" in exc_info.value.message
        assert "test_service" in exc_info.value.message

    def test_circuit_recovery(self):
        """Test circuit recovery after timeout."""
        cb = InfrastructureCircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0,  # Immediate recovery for testing
        )

        call_count = [0]

        @cb
        def sometimes_failing_func():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Failing")
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                sometimes_failing_func()

        assert cb.state == "open"

        # Should recover and succeed
        result = sometimes_failing_func()
        assert result == "success"
        assert cb.state == "closed"

    def test_multiple_successful_calls_reset_count(self):
        """Test successful calls reset failure count."""
        cb = InfrastructureCircuitBreaker(failure_threshold=5)

        call_count = [0]

        @cb
        def sometimes_failing_func():
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise ValueError("Fail on even calls")
            return "success"

        # Success, fail, success pattern
        sometimes_failing_func()  # Success
        with pytest.raises(ValueError):
            sometimes_failing_func()  # Fail

        sometimes_failing_func()  # Success - should reset count

        assert cb.failure_count == 0


class TestDatabaseCircuitBreaker:
    """Test database-specific circuit breaker."""

    def test_create_database_circuit_breaker_defaults(self):
        """Test database circuit breaker with default parameters."""
        cb = create_database_circuit_breaker()

        assert isinstance(cb, InfrastructureCircuitBreaker)
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.name == "database"

    def test_create_database_circuit_breaker_custom(self):
        """Test database circuit breaker with custom parameters."""
        cb = create_database_circuit_breaker(
            failure_threshold=10,
            recovery_timeout=30,
            name="postgres_db",
        )

        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 30
        assert cb.name == "postgres_db"

    def test_database_circuit_breaker_usage(self):
        """Test database circuit breaker in realistic scenario."""
        cb = create_database_circuit_breaker(failure_threshold=3)

        @cb
        def query_database():
            raise Exception("Database connection failed")

        # Simulate database failures
        for _ in range(3):
            with pytest.raises(Exception):
                query_database()

        # Circuit should be open
        with pytest.raises(OnexError) as exc_info:
            query_database()

        assert exc_info.value.code == CoreErrorCode.SERVICE_UNAVAILABLE


class TestKafkaCircuitBreaker:
    """Test Kafka-specific circuit breaker."""

    def test_create_kafka_circuit_breaker_defaults(self):
        """Test Kafka circuit breaker with default parameters."""
        cb = create_kafka_circuit_breaker()

        assert isinstance(cb, InfrastructureCircuitBreaker)
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 30
        assert cb.name == "kafka"

    def test_create_kafka_circuit_breaker_custom(self):
        """Test Kafka circuit breaker with custom parameters."""
        cb = create_kafka_circuit_breaker(
            failure_threshold=5,
            recovery_timeout=15,
            name="kafka_producer",
        )

        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 15
        assert cb.name == "kafka_producer"

    def test_kafka_circuit_breaker_usage(self):
        """Test Kafka circuit breaker in realistic scenario."""
        cb = create_kafka_circuit_breaker(failure_threshold=3)

        @cb
        def send_kafka_message():
            raise Exception("Kafka broker unavailable")

        # Simulate Kafka failures
        for _ in range(3):
            with pytest.raises(Exception):
                send_kafka_message()

        # Circuit should be open
        with pytest.raises(OnexError):
            send_kafka_message()


class TestNetworkCircuitBreaker:
    """Test network-specific circuit breaker."""

    def test_create_network_circuit_breaker_defaults(self):
        """Test network circuit breaker with default parameters."""
        cb = create_network_circuit_breaker()

        assert isinstance(cb, InfrastructureCircuitBreaker)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 45
        assert cb.name == "network"

    def test_create_network_circuit_breaker_custom(self):
        """Test network circuit breaker with custom parameters."""
        cb = create_network_circuit_breaker(
            failure_threshold=5,
            recovery_timeout=20,
            name="external_api",
        )

        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 20
        assert cb.name == "external_api"

    def test_network_circuit_breaker_usage(self):
        """Test network circuit breaker in realistic scenario."""
        cb = create_network_circuit_breaker(failure_threshold=2)

        @cb
        def fetch_external_api():
            raise Exception("Connection timeout")

        # Simulate network failures
        for _ in range(2):
            with pytest.raises(Exception):
                fetch_external_api()

        # Circuit should be open
        with pytest.raises(OnexError):
            fetch_external_api()


class TestVaultCircuitBreaker:
    """Test Vault-specific circuit breaker."""

    def test_create_vault_circuit_breaker_defaults(self):
        """Test Vault circuit breaker with default parameters."""
        cb = create_vault_circuit_breaker()

        assert isinstance(cb, InfrastructureCircuitBreaker)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 90
        assert cb.name == "vault"

    def test_create_vault_circuit_breaker_custom(self):
        """Test Vault circuit breaker with custom parameters."""
        cb = create_vault_circuit_breaker(
            failure_threshold=2,
            recovery_timeout=60,
            name="vault_secrets",
        )

        assert cb.failure_threshold == 2
        assert cb.recovery_timeout == 60
        assert cb.name == "vault_secrets"

    def test_vault_circuit_breaker_usage(self):
        """Test Vault circuit breaker in realistic scenario."""
        cb = create_vault_circuit_breaker(failure_threshold=2)

        @cb
        def get_vault_secret():
            raise Exception("Vault authentication failed")

        # Simulate Vault failures
        for _ in range(2):
            with pytest.raises(Exception):
                get_vault_secret()

        # Circuit should be open
        with pytest.raises(OnexError):
            get_vault_secret()


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker patterns."""

    def test_different_breakers_independent(self):
        """Test that different circuit breakers operate independently."""
        db_breaker = create_database_circuit_breaker(failure_threshold=2)
        kafka_breaker = create_kafka_circuit_breaker(failure_threshold=2)

        @db_breaker
        def db_operation():
            raise Exception("DB error")

        @kafka_breaker
        def kafka_operation():
            return "success"

        # Open database breaker
        for _ in range(2):
            with pytest.raises(Exception):
                db_operation()

        # Kafka breaker should still work
        result = kafka_operation()
        assert result == "success"
        assert db_breaker.state == "open"
        assert kafka_breaker.state == "closed"

    def test_circuit_breaker_with_partial_failures(self):
        """Test circuit breaker with intermittent failures."""
        cb = InfrastructureCircuitBreaker(failure_threshold=3)

        call_count = [0]

        @cb
        def intermittent_func():
            call_count[0] += 1
            if call_count[0] in [2, 4]:
                raise ValueError("Intermittent failure")
            return "success"

        # Pattern: success, fail, success, fail
        intermittent_func()  # Success
        with pytest.raises(ValueError):
            intermittent_func()  # Fail

        intermittent_func()  # Success - resets count
        with pytest.raises(ValueError):
            intermittent_func()  # Fail

        # Circuit should still be closed (failures not consecutive after reset)
        assert cb.state == "closed"

    def test_circuit_breaker_async_compatibility(self):
        """Test circuit breaker with async functions."""
        cb = InfrastructureCircuitBreaker(failure_threshold=2)

        @cb
        async def async_operation():
            return "async_success"

        # Note: Circuit breaker wraps async functions but returns coroutines
        # This test verifies the decorator doesn't break async functions
        import asyncio

        result = asyncio.run(async_operation())
        assert result == "async_success"

    def test_onex_error_chaining(self):
        """Test that OnexError properly chains circuit breaker errors."""
        cb = InfrastructureCircuitBreaker(
            failure_threshold=1,
            name="test_service",
        )

        @cb
        def failing_operation():
            raise Exception("Original error")

        # Open the circuit
        with pytest.raises(Exception):
            failing_operation()

        # Next call should chain the circuit breaker error
        with pytest.raises(OnexError) as exc_info:
            failing_operation()

        error = exc_info.value
        assert error.code == CoreErrorCode.SERVICE_UNAVAILABLE
        assert "Circuit breaker open" in error.message
        assert error.__cause__ is not None  # Should have chained exception

    def test_multiple_decorators(self):
        """Test circuit breaker with multiple decorators."""
        cb1 = InfrastructureCircuitBreaker(failure_threshold=2, name="breaker1")
        cb2 = InfrastructureCircuitBreaker(failure_threshold=2, name="breaker2")

        @cb1
        @cb2
        def multi_decorated():
            raise ValueError("Test error")

        # Both breakers should track failures
        for _ in range(2):
            with pytest.raises(ValueError):
                multi_decorated()

        # Both breakers should be open
        with pytest.raises(OnexError):
            multi_decorated()
