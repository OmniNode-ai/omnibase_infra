# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dead Letter Queue (DLQ) recovery tests for OMN-955.

This test suite validates DLQ behavior for permanently failing messages.
It tests:

1. Messages move to DLQ after max retries exhausted
2. DLQ message format includes failure context
3. DLQ captures allow later analysis and replay
4. Original message is preserved in DLQ

Architecture:
    The DLQ captures messages that cannot be processed successfully after
    all retry attempts are exhausted. This provides:

    - Failure isolation: Bad messages don't block good ones
    - Auditability: Failed messages can be analyzed
    - Replayability: Messages can be reprocessed after fixing issues

    DLQ Message Format:
    {
        "original_message": {...},          # Original message payload
        "failure_context": {
            "error_type": "...",            # Exception type
            "error_message": "...",         # Error message
            "retry_count": N,               # Number of retries attempted
            "first_failure_at": "...",      # Timestamp of first failure
            "last_failure_at": "...",       # Timestamp of final failure
            "correlation_id": "...",        # Correlation ID for tracing
        },
        "dlq_metadata": {
            "sent_to_dlq_at": "...",        # When message was moved to DLQ
            "source_topic": "...",          # Original topic
            "handler_id": "...",            # Handler that failed
        }
    }

Test Organization:
    - TestDLQCapture: Message capture after max retries
    - TestDLQMessageFormat: DLQ message format validation
    - TestDLQPreservesOriginal: Original message preservation
    - TestDLQMetadata: Failure context and metadata

Related:
    - OMN-955: Failure recovery tests
    - OMN-954: Effect retry and backoff
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Sample Message Models (Typed fixtures for testing)
# =============================================================================


class ModelSampleMessagePayload(BaseModel):
    """Typed payload for sample test messages.

    This model provides strong typing for the payload portion of sample
    messages used in DLQ testing. Following ONEX guidelines, we use
    explicit types rather than `Any`.
    """

    model_config = ConfigDict(frozen=True)

    data: str = Field(description="Test data string")
    value: int = Field(description="Test numeric value")


class ModelSampleMessage(BaseModel):
    """Typed sample message for DLQ testing.

    This model represents a standard event message structure used
    in DLQ tests. All fields are explicitly typed per ONEX requirements.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(description="Unique event identifier")
    event_type: str = Field(description="Event type identifier")
    payload: ModelSampleMessagePayload = Field(description="Event payload")
    timestamp: str = Field(description="ISO-8601 formatted timestamp")


# =============================================================================
# DLQ Models
# =============================================================================


class ModelFailureContext(BaseModel):
    """Failure context for DLQ messages.

    Captures information about why the message failed and how many
    retries were attempted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    error_type: str = Field(description="Exception class name")
    error_message: str = Field(description="Error message")
    retry_count: int = Field(description="Number of retries attempted")
    first_failure_at: datetime = Field(description="Timestamp of first failure")
    last_failure_at: datetime = Field(description="Timestamp of final failure")
    correlation_id: UUID | None = Field(default=None, description="Correlation ID")


class ModelDLQMetadata(BaseModel):
    """Metadata about DLQ capture."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sent_to_dlq_at: datetime = Field(description="When message was moved to DLQ")
    source_topic: str = Field(description="Original topic")
    handler_id: str = Field(description="Handler that failed")
    instance_id: str | None = Field(default=None, description="Instance identifier")


class ModelDLQMessage(BaseModel):
    """Complete DLQ message with original content and failure context.

    Note on `original_message` typing:
        Uses `dict[str, object]` instead of `dict[str, Any]` per ONEX guidelines.
        The `object` type explicitly indicates "any Python object" while satisfying
        the "no Any types" policy. DLQ must preserve arbitrary message payloads
        from different sources, so the value type cannot be more specific.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: UUID = Field(description="Unique DLQ message ID")
    original_message: dict[str, object] = Field(description="Original message payload")
    failure_context: ModelFailureContext = Field(description="Failure information")
    dlq_metadata: ModelDLQMetadata = Field(description="DLQ capture metadata")


# =============================================================================
# Mock DLQ Infrastructure
# =============================================================================


@dataclass
class MockDLQStore:
    """Mock Dead Letter Queue store for testing.

    Provides an in-memory DLQ that captures failed messages and allows
    querying for testing purposes.
    """

    messages: list[ModelDLQMessage] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send_to_dlq(
        self,
        original_message: dict[str, object],
        failure_context: ModelFailureContext,
        dlq_metadata: ModelDLQMetadata,
    ) -> ModelDLQMessage:
        """Send a failed message to the DLQ.

        Args:
            original_message: The original message that failed.
            failure_context: Information about the failure.
            dlq_metadata: Metadata about the DLQ capture.

        Returns:
            The DLQ message that was stored.
        """
        dlq_message = ModelDLQMessage(
            message_id=uuid4(),
            original_message=original_message,
            failure_context=failure_context,
            dlq_metadata=dlq_metadata,
        )

        async with self._lock:
            self.messages.append(dlq_message)

        return dlq_message

    async def get_messages(
        self,
        source_topic: str | None = None,
        handler_id: str | None = None,
        limit: int | None = None,
    ) -> list[ModelDLQMessage]:
        """Query messages from the DLQ.

        Args:
            source_topic: Filter by source topic.
            handler_id: Filter by handler ID.
            limit: Maximum number of messages to return.

        Returns:
            List of matching DLQ messages.
        """
        async with self._lock:
            result = list(self.messages)

        if source_topic:
            result = [m for m in result if m.dlq_metadata.source_topic == source_topic]

        if handler_id:
            result = [m for m in result if m.dlq_metadata.handler_id == handler_id]

        if limit:
            result = result[:limit]

        return result

    async def get_message_count(self) -> int:
        """Get total number of messages in DLQ."""
        async with self._lock:
            return len(self.messages)

    async def clear(self) -> None:
        """Clear all messages from DLQ."""
        async with self._lock:
            self.messages.clear()


# =============================================================================
# Message Processor with DLQ Support
# =============================================================================


class MessageProcessorWithDLQ:
    """Message processor with retry logic and DLQ support.

    This class simulates a message processor that attempts to process
    messages with retries and sends permanently failing messages to DLQ.

    Attributes:
        dlq_store: DLQ store for failed messages.
        max_retries: Maximum number of retry attempts.
        handler_id: Identifier for this handler.
        processed_count: Number of successfully processed messages.
        dlq_count: Number of messages sent to DLQ.
    """

    def __init__(
        self,
        dlq_store: MockDLQStore,
        max_retries: int = 3,
        handler_id: str = "test-handler",
    ) -> None:
        """Initialize message processor.

        Args:
            dlq_store: DLQ store for failed messages.
            max_retries: Maximum retry attempts before DLQ.
            handler_id: Handler identifier for DLQ metadata.
        """
        self.dlq_store = dlq_store
        self.max_retries = max_retries
        self.handler_id = handler_id
        self.processed_count = 0
        self.dlq_count = 0
        self._lock = asyncio.Lock()

    async def process_message(
        self,
        message: dict[str, object],
        source_topic: str = "test-topic",
        should_fail: bool = False,
        fail_count: int | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Process a message with retry logic and DLQ.

        Args:
            message: Message to process.
            source_topic: Topic the message came from.
            should_fail: If True, message processing always fails.
            fail_count: If set, fail exactly this many times then succeed.
            correlation_id: Correlation ID for tracing.

        Returns:
            True if message was processed successfully, False if sent to DLQ.
        """
        first_failure_at: datetime | None = None
        last_error: Exception | None = None
        attempts = 0

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                # Simulate processing
                if should_fail:
                    raise ValueError("Message processing failed")

                if fail_count is not None and attempt < fail_count:
                    raise ValueError(f"Transient failure (attempt {attempt + 1})")

                # Success
                async with self._lock:
                    self.processed_count += 1
                return True

            except Exception as e:
                last_error = e
                if first_failure_at is None:
                    first_failure_at = datetime.now(UTC)

        # All retries exhausted - send to DLQ
        if last_error is not None and first_failure_at is not None:
            failure_context = ModelFailureContext(
                error_type=type(last_error).__name__,
                error_message=str(last_error),
                retry_count=attempts,
                first_failure_at=first_failure_at,
                last_failure_at=datetime.now(UTC),
                correlation_id=correlation_id,
            )

            dlq_metadata = ModelDLQMetadata(
                sent_to_dlq_at=datetime.now(UTC),
                source_topic=source_topic,
                handler_id=self.handler_id,
            )

            await self.dlq_store.send_to_dlq(
                original_message=message,
                failure_context=failure_context,
                dlq_metadata=dlq_metadata,
            )

            async with self._lock:
                self.dlq_count += 1

        return False


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def dlq_store() -> MockDLQStore:
    """Create a fresh mock DLQ store."""
    return MockDLQStore()


@pytest.fixture
def processor(dlq_store: MockDLQStore) -> MessageProcessorWithDLQ:
    """Create a message processor with DLQ support."""
    return MessageProcessorWithDLQ(
        dlq_store=dlq_store,
        max_retries=3,
        handler_id="test-handler",
    )


@pytest.fixture
def sample_message() -> ModelSampleMessage:
    """Create a sample message for testing.

    Returns a typed ModelSampleMessage following ONEX guidelines.
    Tests use .model_dump() when passing to process_message() to convert
    to dict[str, object] format expected by the processor.
    """
    return ModelSampleMessage(
        event_id=str(uuid4()),
        event_type="test.event",
        payload=ModelSampleMessagePayload(
            data="test_data",
            value=123,
        ),
        timestamp=datetime.now(UTC).isoformat(),
    )


# =============================================================================
# Test Classes
# =============================================================================
# NOTE: correlation_id fixture is provided by chaos/conftest.py


@pytest.mark.unit
@pytest.mark.chaos
class TestDLQCapture:
    """Test DLQ message capture after max retries."""

    @pytest.mark.asyncio
    async def test_message_sent_to_dlq_after_max_retries(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test message is sent to DLQ after exhausting all retries.

        Scenario:
            1. Message processing fails repeatedly
            2. After max_retries (3) + 1 initial attempt = 4 attempts
            3. Message is sent to DLQ
        """
        result = await processor.process_message(
            message=sample_message.model_dump(),
            should_fail=True,
        )

        # Message should have failed
        assert result is False

        # Message should be in DLQ
        count = await dlq_store.get_message_count()
        assert count == 1

        # Processor should track DLQ send
        assert processor.dlq_count == 1
        assert processor.processed_count == 0

    @pytest.mark.asyncio
    async def test_successful_message_not_sent_to_dlq(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test successful message is NOT sent to DLQ."""
        result = await processor.process_message(
            message=sample_message.model_dump(),
            should_fail=False,
        )

        # Message should have succeeded
        assert result is True

        # DLQ should be empty
        count = await dlq_store.get_message_count()
        assert count == 0

        # Processor should track success
        assert processor.processed_count == 1
        assert processor.dlq_count == 0

    @pytest.mark.asyncio
    async def test_transient_failure_with_eventual_success(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test transient failures don't cause DLQ if eventual success.

        Scenario:
            1. First 2 attempts fail
            2. Third attempt succeeds
            3. Message is NOT sent to DLQ
        """
        result = await processor.process_message(
            message=sample_message.model_dump(),
            fail_count=2,  # Fail first 2, succeed on 3rd
        )

        # Message should have eventually succeeded
        assert result is True

        # DLQ should be empty
        count = await dlq_store.get_message_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_failure_on_last_retry_goes_to_dlq(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test failure on last retry attempt still goes to DLQ.

        Scenario:
            1. Fail all 4 attempts (initial + 3 retries)
            2. Message goes to DLQ
        """
        result = await processor.process_message(
            message=sample_message.model_dump(),
            fail_count=4,  # Fail all attempts
        )

        # Message should have failed
        assert result is False

        # Message should be in DLQ
        count = await dlq_store.get_message_count()
        assert count == 1


@pytest.mark.unit
@pytest.mark.chaos
class TestDLQMessageFormat:
    """Test DLQ message format validation."""

    @pytest.mark.asyncio
    async def test_dlq_message_has_required_fields(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test DLQ message contains all required fields."""
        await processor.process_message(
            message=sample_message.model_dump(),
            should_fail=True,
        )

        messages = await dlq_store.get_messages()
        assert len(messages) == 1

        dlq_msg = messages[0]

        # Verify required fields exist
        assert dlq_msg.message_id is not None
        assert dlq_msg.original_message is not None
        assert dlq_msg.failure_context is not None
        assert dlq_msg.dlq_metadata is not None

    @pytest.mark.asyncio
    async def test_failure_context_includes_error_details(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
        correlation_id: UUID,
    ) -> None:
        """Test failure context includes error type and message."""
        await processor.process_message(
            message=sample_message.model_dump(),
            should_fail=True,
            correlation_id=correlation_id,
        )

        messages = await dlq_store.get_messages()
        context = messages[0].failure_context

        # Verify error details
        assert context.error_type == "ValueError"
        assert "failed" in context.error_message.lower()
        assert context.retry_count == 4  # initial + 3 retries
        assert context.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_failure_context_includes_timestamps(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test failure context includes timing information."""
        await processor.process_message(
            message=sample_message.model_dump(),
            should_fail=True,
        )

        messages = await dlq_store.get_messages()
        context = messages[0].failure_context

        # Verify timestamps
        assert context.first_failure_at is not None
        assert context.last_failure_at is not None
        assert context.last_failure_at >= context.first_failure_at

    @pytest.mark.asyncio
    async def test_dlq_metadata_includes_source_info(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test DLQ metadata includes source topic and handler."""
        source_topic = "orders.created"

        await processor.process_message(
            message=sample_message.model_dump(),
            source_topic=source_topic,
            should_fail=True,
        )

        messages = await dlq_store.get_messages()
        metadata = messages[0].dlq_metadata

        # Verify source info
        assert metadata.source_topic == source_topic
        assert metadata.handler_id == "test-handler"
        assert metadata.sent_to_dlq_at is not None


@pytest.mark.unit
@pytest.mark.chaos
class TestDLQPreservesOriginal:
    """Test that original message is preserved in DLQ."""

    @pytest.mark.asyncio
    async def test_original_message_preserved_exactly(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test original message is preserved without modification."""
        message_dict = sample_message.model_dump()
        await processor.process_message(
            message=message_dict,
            should_fail=True,
        )

        messages = await dlq_store.get_messages()
        preserved = messages[0].original_message

        # Verify original is preserved exactly
        assert preserved == message_dict

    @pytest.mark.asyncio
    async def test_complex_message_preserved(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
    ) -> None:
        """Test complex nested message is preserved correctly."""
        complex_message: dict[str, object] = {
            "event_id": str(uuid4()),
            "nested": {
                "level1": {
                    "level2": {
                        "value": "deep",
                    },
                },
            },
            "array": [1, 2, 3, {"key": "value"}],
            "null_field": None,
            "boolean": True,
            "float": 3.14159,
        }

        await processor.process_message(
            message=complex_message,
            should_fail=True,
        )

        messages = await dlq_store.get_messages()
        preserved = messages[0].original_message

        # Verify complex structure is preserved
        assert preserved == complex_message
        # Use nested access via cast for type-safe deep value verification
        nested = cast(dict[str, object], preserved["nested"])
        level1 = cast(dict[str, object], nested["level1"])
        level2 = cast(dict[str, str], level1["level2"])
        assert level2["value"] == "deep"
        assert preserved["array"] == [1, 2, 3, {"key": "value"}]


@pytest.mark.unit
@pytest.mark.chaos
class TestDLQMetadata:
    """Test DLQ failure context and metadata."""

    @pytest.mark.asyncio
    async def test_retry_count_accurate(
        self,
        dlq_store: MockDLQStore,
    ) -> None:
        """Test retry count reflects actual attempts."""
        # Create processor with different retry counts
        for max_retries in [1, 3, 5]:
            dlq = MockDLQStore()
            processor = MessageProcessorWithDLQ(
                dlq_store=dlq,
                max_retries=max_retries,
            )

            # Explicit type annotation for clarity
            test_message: dict[str, object] = {"test": max_retries}
            await processor.process_message(
                message=test_message,
                should_fail=True,
            )

            messages = await dlq.get_messages()
            assert messages[0].failure_context.retry_count == max_retries + 1

    @pytest.mark.asyncio
    async def test_correlation_id_tracked(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
        sample_message: ModelSampleMessage,
    ) -> None:
        """Test correlation ID is tracked through DLQ."""
        correlation_id = uuid4()

        await processor.process_message(
            message=sample_message.model_dump(),
            should_fail=True,
            correlation_id=correlation_id,
        )

        messages = await dlq_store.get_messages()
        assert messages[0].failure_context.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_multiple_handlers_tracked_separately(
        self,
        dlq_store: MockDLQStore,
    ) -> None:
        """Test DLQ correctly tracks messages from different handlers."""
        processor_a = MessageProcessorWithDLQ(
            dlq_store=dlq_store,
            handler_id="handler-a",
        )
        processor_b = MessageProcessorWithDLQ(
            dlq_store=dlq_store,
            handler_id="handler-b",
        )

        # Both handlers fail messages
        msg_a: dict[str, object] = {"from": "a"}
        msg_b: dict[str, object] = {"from": "b"}
        await processor_a.process_message(
            message=msg_a,
            should_fail=True,
        )
        await processor_b.process_message(
            message=msg_b,
            should_fail=True,
        )

        # Query by handler
        a_messages = await dlq_store.get_messages(handler_id="handler-a")
        b_messages = await dlq_store.get_messages(handler_id="handler-b")

        assert len(a_messages) == 1
        assert len(b_messages) == 1
        assert a_messages[0].original_message["from"] == "a"
        assert b_messages[0].original_message["from"] == "b"

    @pytest.mark.asyncio
    async def test_multiple_topics_tracked_separately(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
    ) -> None:
        """Test DLQ correctly tracks messages from different topics."""
        orders_msg: dict[str, object] = {"topic": "orders"}
        payments_msg: dict[str, object] = {"topic": "payments"}
        await processor.process_message(
            message=orders_msg,
            source_topic="orders.created",
            should_fail=True,
        )
        await processor.process_message(
            message=payments_msg,
            source_topic="payments.processed",
            should_fail=True,
        )

        # Query by topic
        order_messages = await dlq_store.get_messages(source_topic="orders.created")
        payment_messages = await dlq_store.get_messages(
            source_topic="payments.processed"
        )

        assert len(order_messages) == 1
        assert len(payment_messages) == 1


@pytest.mark.unit
@pytest.mark.chaos
class TestDLQConcurrency:
    """Test DLQ behavior under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_failures_captured(
        self,
        dlq_store: MockDLQStore,
    ) -> None:
        """Test concurrent failures are all captured in DLQ."""
        processor = MessageProcessorWithDLQ(
            dlq_store=dlq_store,
            max_retries=1,  # Fast failure for test
        )

        # Process many failing messages concurrently
        num_messages = 50
        # Explicit type annotation for list of messages
        messages: list[dict[str, object]] = [{"id": i} for i in range(num_messages)]

        tasks = [
            processor.process_message(message=msg, should_fail=True) for msg in messages
        ]
        await asyncio.gather(*tasks)

        # All should be in DLQ
        count = await dlq_store.get_message_count()
        assert count == num_messages

    @pytest.mark.asyncio
    async def test_dlq_messages_have_unique_ids(
        self,
        processor: MessageProcessorWithDLQ,
        dlq_store: MockDLQStore,
    ) -> None:
        """Test each DLQ message has a unique ID."""
        # Send multiple messages to DLQ
        for i in range(10):
            msg: dict[str, object] = {"id": i}
            await processor.process_message(
                message=msg,
                should_fail=True,
            )

        messages = await dlq_store.get_messages()

        # All message IDs should be unique
        message_ids = [m.message_id for m in messages]
        assert len(message_ids) == len(set(message_ids))


__all__ = [
    # Sample message models
    "ModelSampleMessagePayload",
    "ModelSampleMessage",
    # DLQ models
    "ModelFailureContext",
    "ModelDLQMetadata",
    "ModelDLQMessage",
    # Mock infrastructure
    "MockDLQStore",
    "MessageProcessorWithDLQ",
    # Test classes
    "TestDLQCapture",
    "TestDLQMessageFormat",
    "TestDLQPreservesOriginal",
    "TestDLQMetadata",
    "TestDLQConcurrency",
]
