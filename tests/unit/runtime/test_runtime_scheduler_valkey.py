# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RuntimeScheduler Valkey persistence functionality.

This test suite validates the Valkey persistence features of RuntimeScheduler:
- Lazy client creation with retry logic
- Connection error handling and fallback behavior
- Sequence number loading and persistence
- Exponential backoff during connection retries
- _last_persisted_sequence tracking
- Graceful degradation when Valkey is unavailable

Test Organization:
    - TestValkeyClientCreation: Lazy client creation and retry logic
    - TestValkeyConnectionRetry: Exponential backoff and retry behavior
    - TestSequenceNumberLoading: Loading sequence from Valkey
    - TestSequenceNumberPersistence: Persisting sequence to Valkey
    - TestValkeyErrorRecovery: Error handling and fallback behavior
    - TestValkeyClientCleanup: Client closure and cleanup

Coverage Goals:
    - All Valkey-related code paths in RuntimeScheduler
    - Connection retry logic with exponential backoff
    - Error recovery and graceful degradation
    - _last_persisted_sequence updates correctly

Related Tickets:
    - OMN-953: RuntimeTick scheduler implementation
    - OMN-1059: PR #107 review feedback - missing tests
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from omnibase_infra.runtime.models import ModelRuntimeSchedulerConfig
from omnibase_infra.runtime.runtime_scheduler import RuntimeScheduler

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Mock EventBusKafka for testing.

    Returns:
        AsyncMock configured to simulate EventBusKafka behavior.
    """
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value=None)
    bus.start = AsyncMock(return_value=None)
    bus.stop = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def valkey_enabled_config() -> ModelRuntimeSchedulerConfig:
    """Configuration with Valkey persistence enabled.

    Returns:
        Configuration with persistence enabled and fast settings for tests.
    """
    return ModelRuntimeSchedulerConfig(
        tick_interval_ms=100,
        scheduler_id="test-valkey-scheduler",
        tick_topic="test.runtime.tick.v1",
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout_seconds=1.0,
        max_tick_jitter_ms=0,
        persist_sequence_number=True,  # Enable persistence
        sequence_number_key="test_scheduler_sequence",
        valkey_host="localhost",
        valkey_port=6379,
        valkey_timeout_seconds=1.0,
        valkey_connection_retries=2,
    )


@pytest.fixture
def valkey_disabled_config() -> ModelRuntimeSchedulerConfig:
    """Configuration with Valkey persistence disabled.

    Returns:
        Configuration with persistence disabled.
    """
    return ModelRuntimeSchedulerConfig(
        tick_interval_ms=100,
        scheduler_id="test-scheduler-no-valkey",
        tick_topic="test.runtime.tick.v1",
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout_seconds=1.0,
        max_tick_jitter_ms=0,
        persist_sequence_number=False,  # Disable persistence
    )


@pytest.fixture
def mock_valkey_client() -> MagicMock:
    """Mock Redis/Valkey client.

    Returns:
        MagicMock configured to simulate redis.asyncio.Redis behavior.
    """
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.aclose = AsyncMock(return_value=None)
    return client


# ============================================================================
# Valkey Client Creation Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestValkeyClientCreation:
    """Test lazy Valkey client creation and initialization."""

    async def test_client_not_created_when_persistence_disabled(
        self,
        valkey_disabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test that Valkey client is not created when persistence is disabled."""
        scheduler = RuntimeScheduler(
            config=valkey_disabled_config, event_bus=mock_event_bus
        )

        # Attempt to get client
        client = await scheduler._get_valkey_client()

        assert client is None
        assert scheduler._valkey_client is None

    async def test_client_created_lazily_on_first_use(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that Valkey client is created lazily on first use."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            # Client should not exist yet
            assert scheduler._valkey_client is None

            # Get client for the first time
            client = await scheduler._get_valkey_client()

            # Client should now be created
            assert client is mock_valkey_client
            assert scheduler._valkey_client is mock_valkey_client
            mock_redis.assert_called_once()

    async def test_client_reused_on_subsequent_calls(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that Valkey client is reused on subsequent calls."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            # Get client twice
            client1 = await scheduler._get_valkey_client()
            client2 = await scheduler._get_valkey_client()

            # Should be the same instance
            assert client1 is client2
            # Redis constructor should only be called once
            mock_redis.assert_called_once()

    async def test_client_returns_none_when_marked_unavailable(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test that client returns None when Valkey is marked unavailable."""
        scheduler = RuntimeScheduler(
            config=valkey_enabled_config, event_bus=mock_event_bus
        )

        # Mark Valkey as unavailable
        scheduler._valkey_available = False

        client = await scheduler._get_valkey_client()

        assert client is None


# ============================================================================
# Connection Retry Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestValkeyConnectionRetry:
    """Test Valkey connection retry logic with exponential backoff."""

    async def test_retry_on_connection_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that connection is retried on connection error."""
        # Configure to fail first attempt, succeed on second
        mock_valkey_client.ping = AsyncMock(
            side_effect=[RedisConnectionError("Connection refused"), True]
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                scheduler = RuntimeScheduler(
                    config=valkey_enabled_config, event_bus=mock_event_bus
                )

                client = await scheduler._get_valkey_client()

                # Should succeed after retry
                assert client is mock_valkey_client
                # Should have slept between retries (exponential backoff)
                mock_sleep.assert_called()

    async def test_exponential_backoff_between_retries(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that exponential backoff is applied between retries.

        The backoff formula is: min(1.0 * (2**attempt), 60.0)
        - Attempt 0: 1.0 * 2^0 = 1.0
        - Attempt 1: 1.0 * 2^1 = 2.0
        - Attempt 2: 1.0 * 2^2 = 4.0
        """
        # Configure to fail first two attempts, succeed on third
        mock_valkey_client.ping = AsyncMock(
            side_effect=[
                RedisConnectionError("Connection refused"),
                RedisConnectionError("Connection refused"),
                True,
            ]
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                scheduler = RuntimeScheduler(
                    config=valkey_enabled_config, event_bus=mock_event_bus
                )

                client = await scheduler._get_valkey_client()

                assert client is mock_valkey_client
                # Verify exponential backoff: 1.0 * (2 ** attempt)
                sleep_calls = mock_sleep.call_args_list
                assert len(sleep_calls) >= 2
                # First retry (attempt=0): 1.0 * 2^0 = 1.0
                assert sleep_calls[0][0][0] == 1.0
                # Second retry (attempt=1): 1.0 * 2^1 = 2.0
                assert sleep_calls[1][0][0] == 2.0

    async def test_marks_unavailable_after_all_retries_exhausted(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that Valkey is marked unavailable after all retries fail."""
        # Configure all attempts to fail
        mock_valkey_client.ping = AsyncMock(
            side_effect=RedisConnectionError("Connection refused")
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client
            with patch("asyncio.sleep", new_callable=AsyncMock):
                scheduler = RuntimeScheduler(
                    config=valkey_enabled_config, event_bus=mock_event_bus
                )

                client = await scheduler._get_valkey_client()

                # Should return None after all retries exhausted
                assert client is None
                # Should be marked as unavailable
                assert scheduler._valkey_available is False
                assert scheduler._valkey_client is None

    async def test_retry_on_timeout_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that connection is retried on timeout error."""
        mock_valkey_client.ping = AsyncMock(
            side_effect=[RedisTimeoutError("Timeout"), True]
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client
            with patch("asyncio.sleep", new_callable=AsyncMock):
                scheduler = RuntimeScheduler(
                    config=valkey_enabled_config, event_bus=mock_event_bus
                )

                client = await scheduler._get_valkey_client()

                assert client is mock_valkey_client

    async def test_retry_on_asyncio_timeout_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that connection is retried on asyncio TimeoutError.

        This is distinct from RedisTimeoutError and tests the asyncio.wait_for
        timeout wrapper around the ping operation.
        """
        mock_valkey_client.ping = AsyncMock(
            side_effect=[TimeoutError("asyncio timeout"), True]
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client
            with patch("asyncio.sleep", new_callable=AsyncMock):
                scheduler = RuntimeScheduler(
                    config=valkey_enabled_config, event_bus=mock_event_bus
                )

                client = await scheduler._get_valkey_client()

                assert client is mock_valkey_client

    async def test_marks_unavailable_on_unexpected_redis_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that unexpected Redis error marks Valkey unavailable immediately."""
        mock_valkey_client.ping = AsyncMock(side_effect=RedisError("Unexpected error"))

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            client = await scheduler._get_valkey_client()

            # Should return None immediately (no retries for unexpected errors)
            assert client is None
            assert scheduler._valkey_available is False


# ============================================================================
# Sequence Number Loading Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestSequenceNumberLoading:
    """Test loading sequence number from Valkey."""

    async def test_load_sequence_skipped_when_persistence_disabled(
        self,
        valkey_disabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test that sequence loading is skipped when persistence is disabled."""
        scheduler = RuntimeScheduler(
            config=valkey_disabled_config, event_bus=mock_event_bus
        )

        await scheduler._load_sequence_number()

        # Sequence should remain at 0
        assert scheduler._sequence_number == 0
        assert scheduler._last_persisted_sequence == 0

    async def test_load_sequence_from_valkey_success(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test successful loading of sequence number from Valkey."""
        mock_valkey_client.get = AsyncMock(return_value="42")

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler._load_sequence_number()

            # Sequence should be loaded from Valkey
            assert scheduler._sequence_number == 42
            assert scheduler._last_persisted_sequence == 42
            mock_valkey_client.get.assert_called_once_with(
                valkey_enabled_config.sequence_number_key
            )

    async def test_load_sequence_starts_at_zero_when_key_not_found(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that sequence starts at 0 when key doesn't exist in Valkey."""
        mock_valkey_client.get = AsyncMock(return_value=None)

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler._load_sequence_number()

            # Sequence should remain at 0
            assert scheduler._sequence_number == 0
            assert scheduler._last_persisted_sequence == 0

    async def test_load_sequence_handles_invalid_value(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that invalid sequence value falls back to 0."""
        mock_valkey_client.get = AsyncMock(return_value="not-a-number")

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler._load_sequence_number()

            # Should fall back to 0
            assert scheduler._sequence_number == 0

    async def test_load_sequence_handles_negative_value(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that negative sequence value falls back to 0."""
        mock_valkey_client.get = AsyncMock(return_value="-5")

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler._load_sequence_number()

            # Should fall back to 0 (negative not allowed)
            assert scheduler._sequence_number == 0

    async def test_load_sequence_handles_valkey_unavailable(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that sequence loading gracefully handles Valkey unavailability."""
        mock_valkey_client.ping = AsyncMock(
            side_effect=RedisConnectionError("Connection refused")
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client
            with patch("asyncio.sleep", new_callable=AsyncMock):
                scheduler = RuntimeScheduler(
                    config=valkey_enabled_config, event_bus=mock_event_bus
                )

                await scheduler._load_sequence_number()

                # Should fall back to 0 gracefully
                assert scheduler._sequence_number == 0
                assert scheduler._valkey_available is False

    async def test_load_sequence_handles_connection_error_during_get(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that connection error during get marks Valkey unavailable."""
        mock_valkey_client.get = AsyncMock(
            side_effect=RedisConnectionError("Connection lost")
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler._load_sequence_number()

            # Should fall back to 0 and mark unavailable
            assert scheduler._sequence_number == 0
            assert scheduler._valkey_available is False

    async def test_load_sequence_handles_timeout_error_during_get(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that asyncio TimeoutError during get marks Valkey unavailable.

        This tests the asyncio.wait_for timeout, distinct from RedisTimeoutError.
        """
        mock_valkey_client.get = AsyncMock(
            side_effect=TimeoutError("Operation timed out")
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler._load_sequence_number()

            # Should fall back to 0 and mark unavailable
            assert scheduler._sequence_number == 0
            assert scheduler._valkey_available is False

    async def test_load_sequence_handles_generic_redis_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that generic RedisError during get is handled gracefully."""
        mock_valkey_client.get = AsyncMock(
            side_effect=RedisError("Unknown Redis error")
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler._load_sequence_number()

            # Should fall back to 0 (but not necessarily mark unavailable
            # since generic RedisError doesn't mark unavailable in load)
            assert scheduler._sequence_number == 0


# ============================================================================
# Sequence Number Persistence Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestSequenceNumberPersistence:
    """Test persisting sequence number to Valkey."""

    async def test_persist_sequence_skipped_when_persistence_disabled(
        self,
        valkey_disabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test that sequence persistence is skipped when disabled."""
        scheduler = RuntimeScheduler(
            config=valkey_disabled_config, event_bus=mock_event_bus
        )
        scheduler._sequence_number = 100

        await scheduler._persist_sequence_number()

        # No Valkey interaction should occur
        assert scheduler._valkey_client is None

    async def test_persist_sequence_to_valkey_success(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test successful persistence of sequence number to Valkey."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )
            scheduler._sequence_number = 100

            await scheduler._persist_sequence_number()

            # Should have written to Valkey
            mock_valkey_client.set.assert_called_once_with(
                valkey_enabled_config.sequence_number_key, "100"
            )
            # Last persisted should be updated
            assert scheduler._last_persisted_sequence == 100

    async def test_persist_updates_last_persisted_sequence(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that _last_persisted_sequence is updated correctly."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            # Simulate emitting ticks
            scheduler._sequence_number = 50

            await scheduler._persist_sequence_number()

            assert scheduler._last_persisted_sequence == 50

            # Emit more ticks
            scheduler._sequence_number = 75

            await scheduler._persist_sequence_number()

            assert scheduler._last_persisted_sequence == 75

    async def test_persist_handles_valkey_unavailable_gracefully(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test that persistence handles Valkey unavailability gracefully.

        When Valkey is unavailable, persistence should:
        - Not raise an exception (graceful fallback)
        - Log a warning message
        - NOT update _last_persisted_sequence (since persistence actually failed)
        """
        scheduler = RuntimeScheduler(
            config=valkey_enabled_config, event_bus=mock_event_bus
        )
        scheduler._sequence_number = 100
        scheduler._valkey_available = False  # Mark as unavailable

        # Should not raise - graceful fallback
        await scheduler._persist_sequence_number()

        # Last persisted should NOT be updated since persistence failed
        # This correctly reflects that the sequence was never persisted
        assert scheduler._last_persisted_sequence == 0

    async def test_persist_handles_connection_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that connection error during persistence is handled.

        When a connection error occurs during persistence:
        - Should not raise an exception (graceful fallback)
        - Should mark Valkey as unavailable
        - Should NOT update _last_persisted_sequence (persistence failed)
        """
        mock_valkey_client.set = AsyncMock(
            side_effect=RedisConnectionError("Connection lost")
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )
            scheduler._sequence_number = 100

            # Should not raise
            await scheduler._persist_sequence_number()

            # Should mark unavailable
            assert scheduler._valkey_available is False
            # Last persisted should NOT be updated since persistence failed
            # This correctly reflects that the sequence was never persisted
            assert scheduler._last_persisted_sequence == 0

    async def test_persist_handles_timeout_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that asyncio TimeoutError during persistence is handled.

        This tests the asyncio.wait_for timeout, distinct from RedisTimeoutError.
        When a timeout occurs during persistence:
        - Should not raise an exception (graceful fallback)
        - Should mark Valkey as unavailable
        - Should NOT update _last_persisted_sequence (persistence failed)
        """
        mock_valkey_client.set = AsyncMock(
            side_effect=TimeoutError("Operation timed out")
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )
            scheduler._sequence_number = 100

            # Should not raise
            await scheduler._persist_sequence_number()

            # Should mark unavailable
            assert scheduler._valkey_available is False
            # Last persisted should NOT be updated since persistence failed
            assert scheduler._last_persisted_sequence == 0

    async def test_persist_handles_redis_timeout_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that RedisTimeoutError during persistence is handled.

        When a Redis timeout occurs during persistence:
        - Should not raise an exception (graceful fallback)
        - Should mark Valkey as unavailable
        - Should NOT update _last_persisted_sequence (persistence failed)
        """
        mock_valkey_client.set = AsyncMock(side_effect=RedisTimeoutError("Timeout"))

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )
            scheduler._sequence_number = 100

            # Should not raise
            await scheduler._persist_sequence_number()

            # Should mark unavailable
            assert scheduler._valkey_available is False
            # Last persisted should NOT be updated since persistence failed
            assert scheduler._last_persisted_sequence == 0

    async def test_persist_handles_generic_redis_error(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that generic RedisError during persistence is handled.

        When a generic Redis error occurs during persistence:
        - Should not raise an exception (graceful fallback)
        - Should NOT update _last_persisted_sequence (persistence failed)
        """
        mock_valkey_client.set = AsyncMock(side_effect=RedisError("Unknown error"))

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )
            scheduler._sequence_number = 100

            # Should not raise
            await scheduler._persist_sequence_number()

            # Last persisted should NOT be updated since persistence failed
            assert scheduler._last_persisted_sequence == 0

    async def test_persist_closes_client_on_completion(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that Valkey client is closed after persistence."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )
            scheduler._sequence_number = 100

            await scheduler._persist_sequence_number()

            # Client should be closed
            mock_valkey_client.aclose.assert_called_once()


# ============================================================================
# Valkey Client Cleanup Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestValkeyClientCleanup:
    """Test Valkey client cleanup and closure."""

    async def test_close_client_when_exists(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that client is closed when it exists."""
        scheduler = RuntimeScheduler(
            config=valkey_enabled_config, event_bus=mock_event_bus
        )
        scheduler._valkey_client = mock_valkey_client

        await scheduler._close_valkey_client()

        mock_valkey_client.aclose.assert_called_once()
        assert scheduler._valkey_client is None

    async def test_close_client_when_not_exists(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test that closing non-existent client is a no-op."""
        scheduler = RuntimeScheduler(
            config=valkey_enabled_config, event_bus=mock_event_bus
        )
        assert scheduler._valkey_client is None

        # Should not raise
        await scheduler._close_valkey_client()

        assert scheduler._valkey_client is None

    async def test_close_handles_error_gracefully(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that close errors are handled gracefully."""
        mock_valkey_client.aclose = AsyncMock(side_effect=Exception("Close failed"))

        scheduler = RuntimeScheduler(
            config=valkey_enabled_config, event_bus=mock_event_bus
        )
        scheduler._valkey_client = mock_valkey_client

        # Should not raise
        await scheduler._close_valkey_client()

        # Client reference should still be cleared
        assert scheduler._valkey_client is None

    async def test_double_close_is_handled_gracefully(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that calling _close_valkey_client() twice is safe.

        The atomic swap pattern in _close_valkey_client() should prevent
        double-close issues even with concurrent coroutine access.
        """
        scheduler = RuntimeScheduler(
            config=valkey_enabled_config, event_bus=mock_event_bus
        )
        scheduler._valkey_client = mock_valkey_client

        # First close
        await scheduler._close_valkey_client()
        assert scheduler._valkey_client is None
        assert mock_valkey_client.aclose.call_count == 1

        # Second close should be a no-op (client already None)
        await scheduler._close_valkey_client()
        assert scheduler._valkey_client is None
        # aclose should NOT be called again
        assert mock_valkey_client.aclose.call_count == 1

    async def test_concurrent_close_is_safe(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that concurrent _close_valkey_client() calls are safe.

        Due to the atomic swap pattern, even if multiple coroutines call
        close simultaneously, aclose should only be called once.
        """
        scheduler = RuntimeScheduler(
            config=valkey_enabled_config, event_bus=mock_event_bus
        )
        scheduler._valkey_client = mock_valkey_client

        # Call close concurrently from multiple coroutines
        await asyncio.gather(
            scheduler._close_valkey_client(),
            scheduler._close_valkey_client(),
            scheduler._close_valkey_client(),
        )

        # Client should be None
        assert scheduler._valkey_client is None
        # aclose should only be called once due to atomic swap
        assert mock_valkey_client.aclose.call_count == 1


# ============================================================================
# Integration with Lifecycle Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestValkeyLifecycleIntegration:
    """Test Valkey integration with scheduler lifecycle."""

    async def test_start_loads_sequence_from_valkey(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that start() loads sequence number from Valkey."""
        mock_valkey_client.get = AsyncMock(return_value="100")

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler.start()

            # Sequence should be loaded
            assert scheduler._sequence_number == 100

            await scheduler.stop()

    async def test_stop_persists_sequence_to_valkey(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that stop() persists sequence number to Valkey."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler.start()

            # Emit some ticks
            await scheduler.emit_tick()
            await scheduler.emit_tick()
            await scheduler.emit_tick()

            await scheduler.stop()

            # Should have persisted sequence 3
            mock_valkey_client.set.assert_called_with(
                valkey_enabled_config.sequence_number_key, "3"
            )

    async def test_stop_closes_valkey_client(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that stop() closes Valkey client."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler.start()
            await scheduler.stop()

            # Client should be closed
            mock_valkey_client.aclose.assert_called()

    async def test_graceful_degradation_when_valkey_unavailable(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that scheduler degrades gracefully when Valkey is unavailable."""
        mock_valkey_client.ping = AsyncMock(
            side_effect=RedisConnectionError("Connection refused")
        )

        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client
            with patch("asyncio.sleep", new_callable=AsyncMock):
                scheduler = RuntimeScheduler(
                    config=valkey_enabled_config, event_bus=mock_event_bus
                )

                # Start should succeed even if Valkey is unavailable
                await scheduler.start()

                assert scheduler.is_running is True

                # Emit ticks should work
                await scheduler.emit_tick()
                assert scheduler._sequence_number == 1

                await scheduler.stop()


# ============================================================================
# Metrics Integration Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestValkeyMetricsIntegration:
    """Test Valkey persistence metrics."""

    async def test_metrics_track_last_persisted_sequence(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that metrics include last_persisted_sequence."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler.start()

            # Emit ticks
            for _ in range(5):
                await scheduler.emit_tick()

            await scheduler.stop()

            # Check metrics
            metrics = await scheduler.get_metrics()
            assert metrics.current_sequence_number == 5
            assert metrics.last_persisted_sequence == 5

    async def test_unpersisted_sequence_count_accurate(
        self,
        valkey_enabled_config: ModelRuntimeSchedulerConfig,
        mock_event_bus: AsyncMock,
        mock_valkey_client: MagicMock,
    ) -> None:
        """Test that unpersisted_sequence_count is calculated correctly."""
        with patch(
            "omnibase_infra.runtime.runtime_scheduler.redis.Redis"
        ) as mock_redis:
            mock_redis.return_value = mock_valkey_client

            scheduler = RuntimeScheduler(
                config=valkey_enabled_config, event_bus=mock_event_bus
            )

            await scheduler.start()

            # Emit some ticks (but don't stop yet)
            for _ in range(10):
                await scheduler.emit_tick()

            # Check metrics - nothing persisted yet
            metrics = await scheduler.get_metrics()
            assert metrics.current_sequence_number == 10
            # Before stop, last_persisted should still be 0
            assert metrics.last_persisted_sequence == 0
            assert metrics.unpersisted_sequence_count() == 10

            await scheduler.stop()

            # After stop, should be synced
            metrics = await scheduler.get_metrics()
            assert metrics.unpersisted_sequence_count() == 0


__all__: list[str] = []
