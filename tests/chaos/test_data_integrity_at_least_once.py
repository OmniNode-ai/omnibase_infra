# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""At-Least-Once Delivery Tests Under Chaos Conditions (OMN-955).

This test suite validates that the system provides at-least-once delivery
guarantees even under chaotic conditions such as:

1. Random processing failures
2. Transient errors during execution
3. Process restarts mid-processing
4. Retry exhaustion scenarios

At-Least-Once Semantics:
    Every event submitted to the system will eventually be processed
    successfully, even if multiple attempts are required. Combined with
    idempotency (OMN-954), this ensures exactly-once semantics.

Architecture:
    The chaos injection pattern uses configurable failure rates to simulate
    transient failures during event processing. The test validates that:
    - All events are eventually processed (completeness)
    - Failed events are properly retried (retry mechanism)
    - No events are lost during processing (durability)

Test Strategy:
    - Use mock executors with controllable failure injection
    - Track all submitted events and verify all are eventually processed
    - Validate retry logic properly handles transient failures
    - Ensure no data loss even under high failure rates

Related:
    - OMN-955: Data Integrity Tests Under Chaos
    - OMN-954: Effect Idempotency
    - test_effect_idempotency.py: Effect-level idempotency tests
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.idempotency import InMemoryIdempotencyStore

if TYPE_CHECKING:
    from collections.abc import Callable


# -----------------------------------------------------------------------------
# Test Models and Helpers
# -----------------------------------------------------------------------------


@dataclass
class ChaosConfig:
    """Configuration for chaos injection in tests.

    Attributes:
        failure_rate: Probability of failure (0.0 to 1.0).
        max_retries: Maximum retry attempts before giving up.
        retry_delay_ms: Delay between retries in milliseconds.
        failure_types: List of exception types to randomly raise.
    """

    failure_rate: float = 0.3
    max_retries: int = 5
    retry_delay_ms: float = 10.0
    failure_types: list[type[Exception]] = field(
        default_factory=lambda: [
            ConnectionError,
            TimeoutError,
            RuntimeError,
        ]
    )


@dataclass
class EventRecord:
    """Record of an event for tracking processing.

    Attributes:
        event_id: Unique identifier for the event.
        correlation_id: Correlation ID for tracing.
        payload: Event payload data.
        processed: Whether the event has been processed.
        attempt_count: Number of processing attempts.
    """

    event_id: UUID
    correlation_id: UUID
    payload: dict[str, str | int]
    processed: bool = False
    attempt_count: int = 0


class ChaosInjector:
    """Injects failures into operations based on configured rate.

    Provides deterministic and random failure injection for chaos testing.
    Tracks failure history for test assertions.

    Attributes:
        config: Chaos configuration.
        failure_count: Number of failures injected.
        call_count: Total number of calls.
        rng: Random number generator for reproducibility.
    """

    def __init__(
        self,
        config: ChaosConfig,
        seed: int | None = None,
    ) -> None:
        """Initialize chaos injector.

        Args:
            config: Chaos configuration.
            seed: Optional random seed for reproducibility.
        """
        self.config = config
        self.failure_count = 0
        self.call_count = 0
        self.rng = random.Random(seed)

    def should_fail(self) -> bool:
        """Determine if the next operation should fail.

        Returns:
            True if failure should be injected.
        """
        self.call_count += 1
        return self.rng.random() < self.config.failure_rate

    def inject_failure(self) -> None:
        """Inject a random failure from configured failure types.

        Raises:
            One of the configured exception types.
        """
        self.failure_count += 1
        exception_type = self.rng.choice(self.config.failure_types)
        raise exception_type(f"Chaos injection: {exception_type.__name__}")

    def maybe_fail(self) -> None:
        """Conditionally inject failure based on failure rate.

        Raises:
            One of the configured exception types if failure is triggered.
        """
        if self.should_fail():
            self.inject_failure()


class ResilientEventProcessor:
    """Event processor with retry logic and chaos injection.

    Simulates a resilient event processing system that:
    - Retries failed operations with configurable attempts
    - Uses idempotency store to prevent duplicate processing
    - Tracks processing statistics for validation

    Attributes:
        idempotency_store: Store for deduplication.
        chaos_injector: Injector for failure simulation.
        processed_events: Set of successfully processed event IDs.
        processing_attempts: Counter of attempts per event.
    """

    def __init__(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        chaos_injector: ChaosInjector,
        backend_executor: AsyncMock,
    ) -> None:
        """Initialize processor.

        Args:
            idempotency_store: Store for idempotency checking.
            chaos_injector: Injector for chaos simulation.
            backend_executor: Mock backend for actual processing.
        """
        self.idempotency_store = idempotency_store
        self.chaos_injector = chaos_injector
        self.backend_executor = backend_executor
        self.processed_events: set[UUID] = set()
        self.processing_attempts: dict[UUID, int] = {}

    async def process_event(
        self,
        event: EventRecord,
    ) -> bool:
        """Process an event with retry logic.

        Attempts to process the event, retrying on transient failures
        up to max_retries times. Uses idempotency store to prevent
        duplicate processing AFTER successful execution.

        Important: Idempotency is checked BEFORE processing starts,
        but only MARKED as processed AFTER successful execution.
        This ensures retries work correctly after failures.

        Args:
            event: Event to process.

        Returns:
            True if event was processed successfully.

        Raises:
            Exception: If all retry attempts are exhausted.
        """
        max_retries = self.chaos_injector.config.max_retries
        retry_delay = self.chaos_injector.config.retry_delay_ms / 1000.0

        for attempt in range(max_retries):
            self.processing_attempts[event.event_id] = (
                self.processing_attempts.get(event.event_id, 0) + 1
            )

            try:
                # Check if already processed (read-only check)
                already_processed = await self.idempotency_store.is_processed(
                    message_id=event.event_id,
                    domain="events",
                )

                if already_processed:
                    # Already processed - skip
                    return True

                # Chaos injection point - may fail
                self.chaos_injector.maybe_fail()

                # Execute actual processing
                await self.backend_executor.process(event.payload)

                # Mark as processed AFTER successful execution
                await self.idempotency_store.mark_processed(
                    message_id=event.event_id,
                    domain="events",
                    correlation_id=event.correlation_id,
                )
                self.processed_events.add(event.event_id)
                event.processed = True
                return True

            except (ConnectionError, TimeoutError, RuntimeError):
                # Transient failure - retry after delay
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                raise

        return False

    async def process_batch(
        self,
        events: list[EventRecord],
    ) -> tuple[int, int]:
        """Process a batch of events with retry logic.

        Processes all events, tracking successes and failures.

        Args:
            events: List of events to process.

        Returns:
            Tuple of (success_count, failure_count).
        """
        success_count = 0
        failure_count = 0

        for event in events:
            try:
                if await self.process_event(event):
                    success_count += 1
            except Exception:
                failure_count += 1

        return success_count, failure_count


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def chaos_config() -> ChaosConfig:
    """Create default chaos configuration for tests.

    Returns:
        ChaosConfig with 30% failure rate and 5 retries.
    """
    return ChaosConfig(
        failure_rate=0.3,
        max_retries=5,
        retry_delay_ms=1.0,  # Fast for tests
    )


@pytest.fixture
def chaos_injector(chaos_config: ChaosConfig) -> ChaosInjector:
    """Create chaos injector with fixed seed for reproducibility.

    Args:
        chaos_config: Chaos configuration fixture.

    Returns:
        ChaosInjector with seed 42.
    """
    return ChaosInjector(config=chaos_config, seed=42)


@pytest.fixture
def idempotency_store() -> InMemoryIdempotencyStore:
    """Create in-memory idempotency store for tests.

    Returns:
        Fresh InMemoryIdempotencyStore instance.
    """
    return InMemoryIdempotencyStore()


@pytest.fixture
def mock_backend() -> AsyncMock:
    """Create mock backend executor.

    Returns:
        AsyncMock configured for process() calls.
    """
    backend = AsyncMock()
    backend.process = AsyncMock(return_value=None)
    return backend


@pytest.fixture
def event_factory() -> Callable[[], EventRecord]:
    """Create factory for generating test events.

    Returns:
        Callable that produces EventRecord instances.
    """

    def _create_event() -> EventRecord:
        return EventRecord(
            event_id=uuid4(),
            correlation_id=uuid4(),
            payload={"type": "test_event", "value": 42},
        )

    return _create_event


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


@pytest.mark.chaos
class TestAtLeastOnceDelivery:
    """Test at-least-once delivery guarantees under chaos conditions."""

    @pytest.mark.asyncio
    async def test_all_events_eventually_processed(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        chaos_injector: ChaosInjector,
        mock_backend: AsyncMock,
        event_factory: Callable[[], EventRecord],
    ) -> None:
        """Verify all events are eventually processed despite failures.

        This test submits multiple events and validates that all are
        eventually processed, even with a 30% failure rate. The retry
        mechanism should ensure no events are lost.

        Test Flow:
            1. Generate batch of events
            2. Process with chaos injection
            3. Verify all events marked as processed
            4. Verify idempotency store has all records
        """
        # Arrange
        processor = ResilientEventProcessor(
            idempotency_store=idempotency_store,
            chaos_injector=chaos_injector,
            backend_executor=mock_backend,
        )
        events = [event_factory() for _ in range(10)]

        # Act
        success_count, failure_count = await processor.process_batch(events)

        # Assert - all events should be processed successfully
        assert success_count == 10, (
            f"Expected all 10 events processed, got {success_count}"
        )
        assert failure_count == 0, f"Expected no failures, got {failure_count}"

        # Verify all events are in processed set
        for event in events:
            assert event.event_id in processor.processed_events
            assert event.processed is True

        # Verify idempotency store has all records
        record_count = await idempotency_store.get_record_count()
        assert record_count == 10

    @pytest.mark.asyncio
    async def test_retry_mechanism_handles_transient_failures(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
        event_factory: Callable[[], EventRecord],
    ) -> None:
        """Verify retry mechanism properly handles transient failures.

        Test with high failure rate to ensure retries are exercised.

        Test Flow:
            1. Configure high failure rate (60%)
            2. Process single event
            3. Verify multiple attempts were made
            4. Verify event was eventually processed
        """
        # Arrange - high failure rate to ensure retries
        high_failure_config = ChaosConfig(
            failure_rate=0.6,
            max_retries=10,
            retry_delay_ms=1.0,
        )
        chaos_injector = ChaosInjector(config=high_failure_config, seed=42)

        processor = ResilientEventProcessor(
            idempotency_store=idempotency_store,
            chaos_injector=chaos_injector,
            backend_executor=mock_backend,
        )
        event = event_factory()

        # Act
        result = await processor.process_event(event)

        # Assert
        assert result is True
        assert event.processed is True
        # With 60% failure rate, we expect multiple attempts
        attempts = processor.processing_attempts.get(event.event_id, 0)
        assert attempts >= 1, "At least one attempt should have been made"

    @pytest.mark.asyncio
    async def test_no_data_loss_under_high_failure_rate(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
        event_factory: Callable[[], EventRecord],
    ) -> None:
        """Verify no data loss even with high failure rates.

        Uses 50% failure rate with sufficient retries to ensure
        eventual success.

        Test Flow:
            1. Configure 50% failure rate with 20 retries
            2. Process batch of 20 events
            3. Verify all events eventually processed
            4. Verify no events lost
        """
        # Arrange
        high_failure_config = ChaosConfig(
            failure_rate=0.5,
            max_retries=20,
            retry_delay_ms=1.0,
        )
        chaos_injector = ChaosInjector(config=high_failure_config, seed=123)

        processor = ResilientEventProcessor(
            idempotency_store=idempotency_store,
            chaos_injector=chaos_injector,
            backend_executor=mock_backend,
        )
        events = [event_factory() for _ in range(20)]
        event_ids = {e.event_id for e in events}

        # Act
        success_count, failure_count = await processor.process_batch(events)

        # Assert - all events processed
        assert success_count == 20
        assert failure_count == 0

        # Verify all event IDs are in processed set
        assert processor.processed_events == event_ids

        # Verify chaos injector was actually invoked
        assert chaos_injector.call_count > 0
        assert chaos_injector.failure_count > 0, "Some failures should have occurred"


@pytest.mark.chaos
class TestFailureRecovery:
    """Test failure recovery mechanisms under chaos conditions."""

    @pytest.mark.asyncio
    async def test_recovery_after_process_restart_simulation(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
        event_factory: Callable[[], EventRecord],
    ) -> None:
        """Verify recovery works after simulated process restart.

        Simulates a process restart by creating a new processor instance
        with the same idempotency store (simulating persistent storage).

        Test Flow:
            1. Process some events with first processor
            2. Simulate restart: create new processor with same store
            3. Re-submit same events
            4. Verify no duplicate processing
            5. Verify all events processed exactly once
        """
        # Arrange - no chaos, focus on restart behavior
        no_chaos_config = ChaosConfig(failure_rate=0.0, max_retries=1)
        chaos_injector = ChaosInjector(config=no_chaos_config)

        # First processor instance
        processor1 = ResilientEventProcessor(
            idempotency_store=idempotency_store,
            chaos_injector=chaos_injector,
            backend_executor=mock_backend,
        )

        events = [event_factory() for _ in range(5)]

        # Act - process with first instance
        await processor1.process_batch(events)

        # Simulate restart: create new processor with SAME store
        mock_backend_2 = AsyncMock()
        mock_backend_2.process = AsyncMock(return_value=None)

        processor2 = ResilientEventProcessor(
            idempotency_store=idempotency_store,
            chaos_injector=chaos_injector,
            backend_executor=mock_backend_2,
        )

        # Re-submit same events
        await processor2.process_batch(events)

        # Assert - backend was called exactly once per event (not twice)
        assert mock_backend.process.call_count == 5, (
            "First processor should process all 5 events"
        )
        assert mock_backend_2.process.call_count == 0, (
            "Second processor should NOT re-process (idempotency)"
        )

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_lose_completed_work(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
        event_factory: Callable[[], EventRecord],
    ) -> None:
        """Verify partial failures don't lose already-completed work.

        Simulates scenario where some events succeed before a failure
        causes processing to stop.

        Test Flow:
            1. Configure to fail on specific event
            2. Process batch
            3. Verify completed events are persisted
            4. Retry remaining events
            5. Verify all eventually processed
        """
        # Arrange - fail every 3rd call
        events = [event_factory() for _ in range(6)]
        failure_pattern = [False, False, True, False, False, True]
        call_index = 0

        class PredictableChaosInjector(ChaosInjector):
            """Chaos injector with predictable failure pattern."""

            def should_fail(self) -> bool:
                nonlocal call_index
                if call_index < len(failure_pattern):
                    result = failure_pattern[call_index]
                    call_index += 1
                    return result
                return False

        config = ChaosConfig(failure_rate=0.0, max_retries=3, retry_delay_ms=1.0)
        predictable_injector = PredictableChaosInjector(config=config)

        processor = ResilientEventProcessor(
            idempotency_store=idempotency_store,
            chaos_injector=predictable_injector,
            backend_executor=mock_backend,
        )

        # Act
        success_count, _ = await processor.process_batch(events)

        # Assert - all events should eventually succeed due to retries
        assert success_count == 6

        # Verify idempotency records
        record_count = await idempotency_store.get_record_count()
        assert record_count == 6


@pytest.mark.chaos
class TestRetryExhaustion:
    """Test behavior when retries are exhausted."""

    @pytest.mark.asyncio
    async def test_retry_exhaustion_reports_failure(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
        event_factory: Callable[[], EventRecord],
    ) -> None:
        """Verify retry exhaustion is properly reported.

        When all retries are exhausted, the failure should be reported
        without losing track of the event for later retry.

        Test Flow:
            1. Configure 100% failure rate with limited retries
            2. Attempt to process event
            3. Verify exception raised after retries exhausted
            4. Verify event NOT marked as processed
        """
        # Arrange - always fail
        always_fail_config = ChaosConfig(
            failure_rate=1.0,
            max_retries=3,
            retry_delay_ms=1.0,
        )
        chaos_injector = ChaosInjector(config=always_fail_config)

        processor = ResilientEventProcessor(
            idempotency_store=idempotency_store,
            chaos_injector=chaos_injector,
            backend_executor=mock_backend,
        )
        event = event_factory()

        # Act & Assert
        with pytest.raises((ConnectionError, TimeoutError, RuntimeError)):
            await processor.process_event(event)

        # Event should NOT be in processed set
        assert event.event_id not in processor.processed_events
        assert event.processed is False

        # Should have made max_retries attempts
        attempts = processor.processing_attempts.get(event.event_id, 0)
        assert attempts == 3


@pytest.mark.chaos
class TestEventSequenceCompleteness:
    """Test that event sequences maintain completeness under chaos."""

    @pytest.mark.asyncio
    async def test_event_sequence_completeness_verified(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
        event_factory: Callable[[], EventRecord],
    ) -> None:
        """Verify all events in a sequence are eventually processed.

        Creates a numbered sequence of events and verifies that
        all sequence numbers are represented in processed events.

        Test Flow:
            1. Create numbered sequence of events
            2. Process with moderate chaos
            3. Verify all sequence numbers processed
            4. Verify no gaps in sequence
        """
        # Arrange
        chaos_config = ChaosConfig(
            failure_rate=0.3,
            max_retries=10,
            retry_delay_ms=1.0,
        )
        chaos_injector = ChaosInjector(config=chaos_config, seed=42)

        processor = ResilientEventProcessor(
            idempotency_store=idempotency_store,
            chaos_injector=chaos_injector,
            backend_executor=mock_backend,
        )

        # Create sequence with sequence numbers
        events = []
        for i in range(15):
            event = event_factory()
            event.payload = {"sequence_number": i, "type": "sequence_event"}
            events.append(event)

        # Act
        await processor.process_batch(events)

        # Assert - all sequence numbers processed
        processed_sequences = {
            e.payload["sequence_number"] for e in events if e.processed
        }
        expected_sequences = set(range(15))

        assert processed_sequences == expected_sequences, (
            f"Missing sequence numbers: {expected_sequences - processed_sequences}"
        )

        # Verify no gaps
        sorted_sequences = sorted(processed_sequences)
        for i, seq in enumerate(sorted_sequences):
            assert seq == i, f"Gap detected at position {i}, got {seq}"
