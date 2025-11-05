"""
Comprehensive tests for Retry Policy Factory.

Tests retry policies, exponential backoff, exception handling,
and service-specific configurations.
"""

import time
from unittest.mock import Mock

import pytest
from tenacity import RetryError

from omnibase_infra.infrastructure.resilience.retry_policy import (
    create_database_retry_policy,
    create_kafka_retry_policy,
    create_network_retry_policy,
    create_vault_retry_policy,
)


class TestDatabaseRetryPolicy:
    """Test database-specific retry policy."""

    def test_create_database_retry_policy_defaults(self):
        """Test database retry policy with default parameters."""
        retry_decorator = create_database_retry_policy()

        assert retry_decorator is not None

    def test_database_retry_successful_execution(self):
        """Test successful execution without retries."""
        retry_decorator = create_database_retry_policy()

        @retry_decorator
        def successful_operation():
            return "success"

        result = successful_operation()
        assert result == "success"

    def test_database_retry_on_connection_error(self):
        """Test retry on ConnectionError."""
        retry_decorator = create_database_retry_policy(max_attempts=3)

        call_count = [0]

        @retry_decorator
        def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Database unavailable")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count[0] == 3

    def test_database_retry_on_timeout_error(self):
        """Test retry on TimeoutError."""
        retry_decorator = create_database_retry_policy(max_attempts=2)

        call_count = [0]

        @retry_decorator
        def timeout_then_success():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError("Query timeout")
            return "success"

        result = timeout_then_success()
        assert result == "success"
        assert call_count[0] == 2

    def test_database_retry_exhausted(self):
        """Test retry exhaustion after max attempts."""
        retry_decorator = create_database_retry_policy(max_attempts=3)

        call_count = [0]

        @retry_decorator
        def always_fails():
            call_count[0] += 1
            raise ConnectionError("Permanent failure")

        with pytest.raises(RetryError):
            always_fails()

        assert call_count[0] == 3

    def test_database_retry_no_retry_on_different_exception(self):
        """Test that other exceptions are not retried."""
        retry_decorator = create_database_retry_policy()

        call_count = [0]

        @retry_decorator
        def raises_value_error():
            call_count[0] += 1
            raise ValueError("Not a retryable error")

        with pytest.raises(ValueError):
            raises_value_error()

        # Should not retry on ValueError
        assert call_count[0] == 1

    def test_database_retry_custom_parameters(self):
        """Test database retry with custom parameters."""
        retry_decorator = create_database_retry_policy(
            max_attempts=5,
            min_wait=0.1,
            max_wait=1.0,
        )

        @retry_decorator
        def test_operation():
            return "success"

        result = test_operation()
        assert result == "success"


class TestNetworkRetryPolicy:
    """Test network-specific retry policy."""

    def test_create_network_retry_policy_defaults(self):
        """Test network retry policy with default parameters."""
        retry_decorator = create_network_retry_policy()

        assert retry_decorator is not None

    def test_network_retry_successful_execution(self):
        """Test successful execution without retries."""
        retry_decorator = create_network_retry_policy()

        @retry_decorator
        def fetch_api():
            return {"status": "ok"}

        result = fetch_api()
        assert result["status"] == "ok"

    def test_network_retry_on_connection_error(self):
        """Test retry on network connection error."""
        retry_decorator = create_network_retry_policy(max_attempts=3)

        call_count = [0]

        @retry_decorator
        def flaky_network():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network unreachable")
            return "success"

        result = flaky_network()
        assert result == "success"
        assert call_count[0] == 3

    def test_network_retry_exhausted(self):
        """Test network retry exhaustion."""
        retry_decorator = create_network_retry_policy(max_attempts=2)

        @retry_decorator
        def always_fails():
            raise TimeoutError("Request timeout")

        with pytest.raises(RetryError):
            always_fails()

    def test_network_retry_custom_wait_times(self):
        """Test network retry with custom wait times."""
        retry_decorator = create_network_retry_policy(
            max_attempts=3,
            min_wait=0.05,
            max_wait=0.5,
        )

        call_count = [0]
        start_time = time.time()

        @retry_decorator
        def slow_operation():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = slow_operation()
        elapsed = time.time() - start_time

        assert result == "success"
        assert elapsed < 2.0  # Should complete quickly with short waits


class TestKafkaRetryPolicy:
    """Test Kafka-specific retry policy."""

    def test_create_kafka_retry_policy_defaults(self):
        """Test Kafka retry policy with default parameters."""
        retry_decorator = create_kafka_retry_policy()

        assert retry_decorator is not None

    def test_kafka_retry_successful_execution(self):
        """Test successful Kafka operation without retries."""
        retry_decorator = create_kafka_retry_policy()

        @retry_decorator
        def send_message():
            return "sent"

        result = send_message()
        assert result == "sent"

    def test_kafka_retry_on_connection_error(self):
        """Test retry on Kafka connection error."""
        retry_decorator = create_kafka_retry_policy(max_attempts=4)

        call_count = [0]

        @retry_decorator
        def kafka_send():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Broker unavailable")
            return "sent"

        result = kafka_send()
        assert result == "sent"
        assert call_count[0] == 3

    def test_kafka_retry_exhausted(self):
        """Test Kafka retry exhaustion."""
        retry_decorator = create_kafka_retry_policy(max_attempts=5)

        @retry_decorator
        def always_fails():
            raise ConnectionError("Permanent broker failure")

        with pytest.raises(RetryError):
            always_fails()


class TestVaultRetryPolicy:
    """Test Vault-specific retry policy."""

    def test_create_vault_retry_policy_defaults(self):
        """Test Vault retry policy with default parameters."""
        retry_decorator = create_vault_retry_policy()

        assert retry_decorator is not None

    def test_vault_retry_successful_execution(self):
        """Test successful Vault operation without retries."""
        retry_decorator = create_vault_retry_policy()

        @retry_decorator
        def get_secret():
            return {"password": "secret"}

        result = get_secret()
        assert result["password"] == "secret"

    def test_vault_retry_on_connection_error(self):
        """Test retry on Vault connection error."""
        retry_decorator = create_vault_retry_policy(max_attempts=3)

        call_count = [0]

        @retry_decorator
        def vault_operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Vault service unavailable")
            return "success"

        result = vault_operation()
        assert result == "success"
        assert call_count[0] == 2

    def test_vault_retry_exhausted(self):
        """Test Vault retry exhaustion."""
        retry_decorator = create_vault_retry_policy(max_attempts=2)

        @retry_decorator
        def always_fails():
            raise TimeoutError("Vault timeout")

        with pytest.raises(RetryError):
            always_fails()


class TestRetryPolicyIntegration:
    """Integration tests for retry policies."""

    def test_combined_policies_independent(self):
        """Test that different retry policies work independently."""
        db_retry = create_database_retry_policy(max_attempts=2)
        net_retry = create_network_retry_policy(max_attempts=2)

        db_calls = [0]
        net_calls = [0]

        @db_retry
        def db_operation():
            db_calls[0] += 1
            if db_calls[0] < 2:
                raise ConnectionError("DB error")
            return "db_success"

        @net_retry
        def network_operation():
            net_calls[0] += 1
            return "net_success"

        # Both should execute independently
        db_result = db_operation()
        net_result = network_operation()

        assert db_result == "db_success"
        assert net_result == "net_success"
        assert db_calls[0] == 2
        assert net_calls[0] == 1

    def test_retry_with_state_mutation(self):
        """Test retry with state changes between attempts."""
        retry_decorator = create_database_retry_policy(max_attempts=3)

        state = {"attempts": 0, "data": []}

        @retry_decorator
        def stateful_operation():
            state["attempts"] += 1
            state["data"].append(state["attempts"])

            if state["attempts"] < 3:
                raise ConnectionError("Not yet")

            return state["data"]

        result = stateful_operation()

        assert result == [1, 2, 3]
        assert state["attempts"] == 3

    @pytest.mark.asyncio
    async def test_retry_with_async_function(self):
        """Test retry decorator with async functions."""
        retry_decorator = create_database_retry_policy(max_attempts=2)

        call_count = [0]

        @retry_decorator
        async def async_operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Async failure")
            return "async_success"

        result = await async_operation()
        assert result == "async_success"
        assert call_count[0] == 2

    def test_retry_preserves_function_metadata(self):
        """Test that retry decorator preserves function metadata."""
        retry_decorator = create_database_retry_policy()

        @retry_decorator
        def documented_function():
            """This function has documentation."""
            return "result"

        # Function metadata should be preserved
        assert documented_function.__name__ == "documented_function"
        assert "documentation" in documented_function.__doc__

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff timing is respected."""
        retry_decorator = create_network_retry_policy(
            max_attempts=3,
            min_wait=0.1,
            max_wait=0.5,
        )

        call_times = []

        @retry_decorator
        def timed_operation():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ConnectionError("Not yet")
            return "success"

        result = timed_operation()

        # Verify exponential backoff between calls
        if len(call_times) >= 2:
            first_wait = call_times[1] - call_times[0]
            assert first_wait >= 0.1  # Should wait at least min_wait

        assert result == "success"
