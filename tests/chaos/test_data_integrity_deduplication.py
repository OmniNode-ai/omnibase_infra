# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Deduplication Tests Under Chaos Conditions (OMN-955).

This test suite validates that the deduplication system correctly prevents
double-processing even under chaotic conditions such as:

1. Duplicate message delivery (at-least-once delivery causing duplicates)
2. Concurrent processing of same message
3. Reprocessing after transient failures
4. Process restart with pending messages

Deduplication Semantics:
    Combined with at-least-once delivery, proper deduplication provides
    exactly-once semantics. The InMemoryIdempotencyStore (for testing)
    and PostgresIdempotencyStore (for production) track processed messages
    using a composite key of (domain, message_id).

Architecture:
    Deduplication occurs at multiple levels:
    1. Intent-level: Same intent_id causes no additional execution
    2. Natural key: Same (entity_id, intent_type, registration_id) is deduplicated
    3. Domain isolation: Different domains allow same message_id

Test Strategy:
    - Submit same message multiple times
    - Verify execution count equals 1
    - Test concurrent duplicate submission
    - Validate natural key deduplication

Related:
    - OMN-955: Data Integrity Tests Under Chaos
    - OMN-954: Effect Idempotency
    - test_effect_idempotency.py: Effect-level idempotency tests
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock
from uuid import NAMESPACE_OID, UUID, uuid4, uuid5

import pytest

from omnibase_infra.idempotency import (
    InMemoryIdempotencyStore,
    ModelIdempotencyRecord,
)

# -----------------------------------------------------------------------------
# Test Models and Helpers
# -----------------------------------------------------------------------------


@dataclass
class ProcessingResult:
    """Result of a processing operation.

    Attributes:
        success: Whether processing succeeded.
        was_duplicate: Whether this was detected as duplicate.
        execution_id: ID of the actual execution (for tracking).
    """

    success: bool
    was_duplicate: bool
    execution_id: UUID | None = None


class DeduplicatingProcessor:
    """Processor with deduplication at multiple levels.

    Implements the ONEX deduplication pattern with:
    - Intent-level deduplication (by message_id)
    - Natural key deduplication (by composite key)
    - Domain isolation (different domains are independent)

    Attributes:
        idempotency_store: Store for deduplication tracking.
        backend_executor: Mock backend for actual processing.
        execution_count: Counter for actual backend calls.
        duplicate_count: Counter for detected duplicates.
        execution_log: Log of all execution attempts.
    """

    def __init__(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        backend_executor: AsyncMock,
    ) -> None:
        """Initialize deduplicating processor.

        Args:
            idempotency_store: Store for deduplication.
            backend_executor: Mock backend executor.
        """
        self.idempotency_store = idempotency_store
        self.backend_executor = backend_executor
        self.execution_count = 0
        self.duplicate_count = 0
        self.execution_log: list[dict[str, UUID | str | bool]] = []

    async def process_intent(
        self,
        intent_id: UUID,
        domain: str = "default",
        correlation_id: UUID | None = None,
    ) -> ProcessingResult:
        """Process an intent with intent-level deduplication.

        Checks idempotency store for intent_id before processing.
        Duplicates are detected and logged but do not cause errors.

        Args:
            intent_id: Unique identifier for this intent.
            domain: Domain namespace for isolation.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ProcessingResult with success status and duplicate flag.
        """
        is_new = await self.idempotency_store.check_and_record(
            message_id=intent_id,
            domain=domain,
            correlation_id=correlation_id,
        )

        log_entry: dict[str, UUID | str | bool] = {
            "intent_id": intent_id,
            "domain": domain,
            "is_new": is_new,
        }

        if not is_new:
            # Detected as duplicate
            self.duplicate_count += 1
            log_entry["action"] = "skipped_duplicate"
            self.execution_log.append(log_entry)
            return ProcessingResult(
                success=True,
                was_duplicate=True,
                execution_id=None,
            )

        # New intent - execute
        execution_id = uuid4()
        self.execution_count += 1
        await self.backend_executor.process(intent_id)

        log_entry["action"] = "executed"
        log_entry["execution_id"] = execution_id
        self.execution_log.append(log_entry)

        return ProcessingResult(
            success=True,
            was_duplicate=False,
            execution_id=execution_id,
        )

    async def process_with_natural_key(
        self,
        intent_id: UUID,
        entity_id: UUID,
        intent_type: str,
        registration_id: UUID,
        correlation_id: UUID | None = None,
    ) -> ProcessingResult:
        """Process with natural key deduplication.

        Uses composite key (entity_id, intent_type, registration_id) to
        detect duplicates even when intent_id differs.

        Args:
            intent_id: Unique identifier for this intent.
            entity_id: Entity being processed.
            intent_type: Type of intent.
            registration_id: Registration workflow ID.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ProcessingResult with success status and duplicate flag.
        """
        # Build deterministic UUID from natural key
        natural_key = f"{entity_id}:{intent_type}:{registration_id}"
        natural_key_uuid = uuid5(NAMESPACE_OID, natural_key)

        # Check intent-level deduplication
        is_new_intent = await self.idempotency_store.check_and_record(
            message_id=intent_id,
            domain="intent",
            correlation_id=correlation_id,
        )

        if not is_new_intent:
            self.duplicate_count += 1
            return ProcessingResult(success=True, was_duplicate=True)

        # Check natural key deduplication
        is_new_natural_key = await self.idempotency_store.check_and_record(
            message_id=natural_key_uuid,
            domain="natural_key",
            correlation_id=correlation_id,
        )

        if not is_new_natural_key:
            self.duplicate_count += 1
            return ProcessingResult(success=True, was_duplicate=True)

        # New operation - execute
        execution_id = uuid4()
        self.execution_count += 1
        await self.backend_executor.process(intent_id)

        return ProcessingResult(
            success=True,
            was_duplicate=False,
            execution_id=execution_id,
        )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


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
def processor(
    idempotency_store: InMemoryIdempotencyStore,
    mock_backend: AsyncMock,
) -> DeduplicatingProcessor:
    """Create deduplicating processor for tests.

    Args:
        idempotency_store: Store fixture.
        mock_backend: Backend fixture.

    Returns:
        Configured DeduplicatingProcessor.
    """
    return DeduplicatingProcessor(
        idempotency_store=idempotency_store,
        backend_executor=mock_backend,
    )


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


@pytest.mark.chaos
class TestDuplicateDetection:
    """Test duplicate detection prevents double-processing."""

    @pytest.mark.asyncio
    async def test_duplicate_intent_detected_and_skipped(
        self,
        processor: DeduplicatingProcessor,
        mock_backend: AsyncMock,
    ) -> None:
        """Verify duplicate intent_id is detected and skipped.

        Test Flow:
            1. Process intent with ID X
            2. Process same intent with ID X again
            3. Verify backend called exactly once
            4. Verify second call detected as duplicate
        """
        # Arrange
        intent_id = uuid4()
        correlation_id = uuid4()

        # Act - first processing
        result1 = await processor.process_intent(
            intent_id=intent_id,
            correlation_id=correlation_id,
        )

        # Act - second processing with SAME intent_id
        result2 = await processor.process_intent(
            intent_id=intent_id,
            correlation_id=correlation_id,
        )

        # Assert
        assert result1.success is True
        assert result1.was_duplicate is False
        assert result1.execution_id is not None

        assert result2.success is True
        assert result2.was_duplicate is True
        assert result2.execution_id is None

        assert processor.execution_count == 1
        assert processor.duplicate_count == 1
        mock_backend.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_duplicates_all_detected(
        self,
        processor: DeduplicatingProcessor,
        mock_backend: AsyncMock,
    ) -> None:
        """Verify multiple duplicates are all detected.

        Test Flow:
            1. Submit same intent 10 times
            2. Verify only 1 execution
            3. Verify 9 duplicates detected
        """
        # Arrange
        intent_id = uuid4()

        # Act - submit 10 times
        results = []
        for _ in range(10):
            result = await processor.process_intent(intent_id=intent_id)
            results.append(result)

        # Assert
        assert processor.execution_count == 1
        assert processor.duplicate_count == 9
        mock_backend.process.assert_called_once()

        # First should not be duplicate, rest should be
        assert results[0].was_duplicate is False
        for i, result in enumerate(results[1:], 1):
            assert result.was_duplicate is True, f"Result {i} should be duplicate"

    @pytest.mark.asyncio
    async def test_different_intents_processed_independently(
        self,
        processor: DeduplicatingProcessor,
        mock_backend: AsyncMock,
    ) -> None:
        """Verify different intent_ids are processed independently.

        Test Flow:
            1. Process intent with ID X
            2. Process intent with ID Y
            3. Verify both executed (not treated as duplicates)
        """
        # Arrange
        intent_id_1 = uuid4()
        intent_id_2 = uuid4()

        # Act
        result1 = await processor.process_intent(intent_id=intent_id_1)
        result2 = await processor.process_intent(intent_id=intent_id_2)

        # Assert
        assert result1.was_duplicate is False
        assert result2.was_duplicate is False
        assert processor.execution_count == 2
        assert processor.duplicate_count == 0
        assert mock_backend.process.call_count == 2


@pytest.mark.chaos
class TestNaturalKeyDeduplication:
    """Test natural key deduplication pattern."""

    @pytest.mark.asyncio
    async def test_same_natural_key_different_intent_id_deduplicated(
        self,
        processor: DeduplicatingProcessor,
        mock_backend: AsyncMock,
    ) -> None:
        """Verify same natural key with different intent_id is deduplicated.

        This is the key test for natural key deduplication: even when
        intent_ids differ, the same logical operation (identified by
        natural key) should be treated as duplicate.

        Test Flow:
            1. Process with intent_id=X, natural_key=(A, B, C)
            2. Process with intent_id=Y, natural_key=(A, B, C) <- different ID!
            3. Verify only one backend call
        """
        # Arrange - same natural key components
        entity_id = uuid4()
        intent_type = "consul.register"
        registration_id = uuid4()

        # Different intent IDs
        intent_id_1 = uuid4()
        intent_id_2 = uuid4()

        # Act
        result1 = await processor.process_with_natural_key(
            intent_id=intent_id_1,
            entity_id=entity_id,
            intent_type=intent_type,
            registration_id=registration_id,
        )

        result2 = await processor.process_with_natural_key(
            intent_id=intent_id_2,
            entity_id=entity_id,
            intent_type=intent_type,
            registration_id=registration_id,
        )

        # Assert
        assert result1.was_duplicate is False
        assert result2.was_duplicate is True
        assert processor.execution_count == 1
        mock_backend.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_natural_keys_processed_independently(
        self,
        processor: DeduplicatingProcessor,
        mock_backend: AsyncMock,
    ) -> None:
        """Verify different natural keys are processed independently.

        Test Flow:
            1. Process with natural_key=(A, B, C)
            2. Process with natural_key=(D, B, C) <- different entity_id
            3. Verify both executed
        """
        # Arrange - different entity_ids
        entity_id_1 = uuid4()
        entity_id_2 = uuid4()
        intent_type = "consul.register"
        registration_id = uuid4()

        # Act
        result1 = await processor.process_with_natural_key(
            intent_id=uuid4(),
            entity_id=entity_id_1,
            intent_type=intent_type,
            registration_id=registration_id,
        )

        result2 = await processor.process_with_natural_key(
            intent_id=uuid4(),
            entity_id=entity_id_2,
            intent_type=intent_type,
            registration_id=registration_id,
        )

        # Assert
        assert result1.was_duplicate is False
        assert result2.was_duplicate is False
        assert processor.execution_count == 2


@pytest.mark.chaos
class TestDomainIsolation:
    """Test domain isolation in deduplication."""

    @pytest.mark.asyncio
    async def test_same_intent_different_domains_independent(
        self,
        processor: DeduplicatingProcessor,
        idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Verify same intent_id in different domains is independent.

        This tests the domain isolation feature that allows the same
        message_id to be used across different domains without conflict.

        Test Flow:
            1. Process intent in domain "consul"
            2. Process same intent in domain "postgres"
            3. Verify both executed (different domains)
        """
        # Arrange
        intent_id = uuid4()

        # Act
        result1 = await processor.process_intent(
            intent_id=intent_id,
            domain="consul",
        )

        result2 = await processor.process_intent(
            intent_id=intent_id,
            domain="postgres",
        )

        # Assert
        assert result1.was_duplicate is False
        assert result2.was_duplicate is False
        assert processor.execution_count == 2

        # Both should be in store (different domains)
        record_count = await idempotency_store.get_record_count()
        assert record_count == 2

    @pytest.mark.asyncio
    async def test_duplicate_in_same_domain_detected(
        self,
        processor: DeduplicatingProcessor,
        idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Verify duplicates within same domain are detected.

        Test Flow:
            1. Process intent in domain "consul"
            2. Process same intent in domain "consul" again
            3. Verify second is detected as duplicate
        """
        # Arrange
        intent_id = uuid4()

        # Act
        result1 = await processor.process_intent(
            intent_id=intent_id,
            domain="consul",
        )

        result2 = await processor.process_intent(
            intent_id=intent_id,
            domain="consul",
        )

        # Assert
        assert result1.was_duplicate is False
        assert result2.was_duplicate is True
        assert processor.execution_count == 1

        # Only one record (duplicate not added)
        record_count = await idempotency_store.get_record_count()
        assert record_count == 1


@pytest.mark.chaos
class TestConcurrentDuplicates:
    """Test concurrent duplicate submission handling."""

    @pytest.mark.asyncio
    async def test_concurrent_duplicate_submissions_deduplicated(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
    ) -> None:
        """Verify concurrent duplicate submissions are deduplicated.

        Simulates multiple concurrent submissions of the same message.
        Only one should succeed in processing.

        Test Flow:
            1. Submit same intent concurrently 10 times
            2. Verify only 1 execution
            3. Verify 9 duplicates detected
        """
        # Arrange
        processor = DeduplicatingProcessor(
            idempotency_store=idempotency_store,
            backend_executor=mock_backend,
        )
        intent_id = uuid4()

        # Act - concurrent submissions
        tasks = [processor.process_intent(intent_id=intent_id) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Assert - exactly one non-duplicate
        non_duplicates = [r for r in results if not r.was_duplicate]
        duplicates = [r for r in results if r.was_duplicate]

        assert len(non_duplicates) == 1, (
            f"Expected exactly 1 non-duplicate, got {len(non_duplicates)}"
        )
        assert len(duplicates) == 9, f"Expected 9 duplicates, got {len(duplicates)}"

        assert processor.execution_count == 1
        mock_backend.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_different_intents_all_processed(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
    ) -> None:
        """Verify concurrent different intents are all processed.

        Test Flow:
            1. Submit 10 different intents concurrently
            2. Verify all 10 processed
            3. Verify no duplicates
        """
        # Arrange
        processor = DeduplicatingProcessor(
            idempotency_store=idempotency_store,
            backend_executor=mock_backend,
        )
        intent_ids = [uuid4() for _ in range(10)]

        # Act - concurrent submissions
        tasks = [
            processor.process_intent(intent_id=intent_id) for intent_id in intent_ids
        ]
        results = await asyncio.gather(*tasks)

        # Assert - all should be non-duplicates
        non_duplicates = [r for r in results if not r.was_duplicate]
        assert len(non_duplicates) == 10

        assert processor.execution_count == 10
        assert processor.duplicate_count == 0


@pytest.mark.chaos
class TestIdempotencyRecordPersistence:
    """Test idempotency record persistence and retrieval."""

    @pytest.mark.asyncio
    async def test_idempotency_record_contains_correct_data(
        self,
        idempotency_store: InMemoryIdempotencyStore,
    ) -> None:
        """Verify idempotency records contain correct metadata.

        Test Flow:
            1. Record a message with correlation_id
            2. Retrieve the record
            3. Verify all fields are correct
        """
        # Arrange
        message_id = uuid4()
        correlation_id = uuid4()
        domain = "test_domain"

        # Act
        await idempotency_store.check_and_record(
            message_id=message_id,
            domain=domain,
            correlation_id=correlation_id,
        )

        record = await idempotency_store.get_record(
            message_id=message_id,
            domain=domain,
        )

        # Assert
        assert record is not None
        assert isinstance(record, ModelIdempotencyRecord)
        assert record.message_id == message_id
        assert record.domain == domain
        assert record.correlation_id == correlation_id
        assert record.processed_at is not None

    @pytest.mark.asyncio
    async def test_idempotency_survives_processor_restart(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        mock_backend: AsyncMock,
    ) -> None:
        """Verify idempotency survives processor restart.

        Simulates restart by creating new processor with same store.

        Test Flow:
            1. Process intent with processor 1
            2. Create new processor with same store (simulate restart)
            3. Re-submit same intent
            4. Verify detected as duplicate
        """
        # Arrange
        processor1 = DeduplicatingProcessor(
            idempotency_store=idempotency_store,
            backend_executor=mock_backend,
        )
        intent_id = uuid4()

        # Act - process with first processor
        result1 = await processor1.process_intent(intent_id=intent_id)

        # Simulate restart with new processor
        mock_backend_2 = AsyncMock()
        mock_backend_2.process = AsyncMock(return_value=None)

        processor2 = DeduplicatingProcessor(
            idempotency_store=idempotency_store,
            backend_executor=mock_backend_2,
        )

        # Re-submit same intent
        result2 = await processor2.process_intent(intent_id=intent_id)

        # Assert
        assert result1.was_duplicate is False
        assert result2.was_duplicate is True

        # First backend called, second not
        mock_backend.process.assert_called_once()
        mock_backend_2.process.assert_not_called()
