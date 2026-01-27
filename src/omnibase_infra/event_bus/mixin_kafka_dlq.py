# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dead Letter Queue mixin for Kafka event bus.

This module provides DLQ functionality that can be mixed into EventBusKafka
to handle failed message processing. It supports metrics tracking, callback
hooks for custom alerting, and proper error sanitization.

Features:
    - DLQ message publishing with comprehensive failure metadata
    - Callback registration for custom alerting integration
    - Metrics tracking for monitoring DLQ operations
    - Error sanitization to prevent credential leakage
    - Support for both processed and raw (deserialization failure) messages

Usage:
    ```python
    class EventBusKafka(MixinKafkaDlq, MixinAsyncCircuitBreaker):
        def __init__(self, config):
            # Initialize DLQ mixin
            self._init_dlq()
            # ... rest of init

        # DLQ methods are now available:
        # - register_dlq_callback()
        # - dlq_metrics property
        # - _publish_to_dlq()
        # - _publish_raw_to_dlq()
    ```

Design Note:
    This mixin assumes the parent class has:
    - self._config: ModelKafkaEventBusConfig with dead_letter_topic
    - self._environment: str for environment context
    - self._producer: AIOKafkaProducer | None
    - self._producer_lock: asyncio.Lock for producer access
    - self._timeout_seconds: int for publish timeout
    - self._model_headers_to_kafka(): Method to convert headers
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.event_bus.models import (
    ModelDlqEvent,
    ModelDlqMetrics,
    ModelEventHeaders,
    ModelEventMessage,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from aiokafka import AIOKafkaProducer

    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


@runtime_checkable
class ProtocolKafkaDlqHost(Protocol):
    """Protocol defining methods required by MixinKafkaDlq from its host class.

    This protocol exists to satisfy mypy type checking for mixin classes that
    call methods defined on the parent class (EventBusKafka). It also includes
    attributes defined by the mixin itself that are accessed via self.
    """

    # Attributes from parent class (EventBusKafka)
    _config: ModelKafkaEventBusConfig
    _environment: str
    _producer: AIOKafkaProducer | None
    _producer_lock: asyncio.Lock
    _timeout_seconds: int

    # Attributes defined by the mixin itself
    _dlq_metrics: ModelDlqMetrics
    _dlq_metrics_lock: asyncio.Lock

    def _model_headers_to_kafka(
        self, headers: ModelEventHeaders
    ) -> list[tuple[str, bytes]]:
        """Convert ModelEventHeaders to Kafka header format."""
        ...

    async def _invoke_dlq_callbacks(self, event: ModelDlqEvent) -> None:
        """Invoke registered DLQ callbacks."""
        ...


# Type alias for DLQ callback functions
DlqCallbackType = Callable[[ModelDlqEvent], Awaitable[None]]

logger = logging.getLogger(__name__)


class MixinKafkaDlq:
    """Mixin providing Dead Letter Queue functionality for Kafka event bus.

    This mixin adds DLQ message publishing, callback registration, and metrics
    tracking capabilities to a Kafka event bus implementation.

    Attributes provided by mixin:
        dlq_metrics: Property returning current DLQ metrics (copy-on-write)

    Methods provided by mixin:
        register_dlq_callback: Register callback for DLQ events
        _publish_to_dlq: Publish failed message to DLQ
        _publish_raw_to_dlq: Publish raw message (deserialization failure) to DLQ
        _invoke_dlq_callbacks: Invoke registered callbacks with error isolation

    Required attributes from parent class:
        _config: ModelKafkaEventBusConfig with dead_letter_topic
        _environment: str for environment context
        _group: str for consumer group context
        _producer: AIOKafkaProducer | None
        _producer_lock: asyncio.Lock for producer access
        _timeout_seconds: int for publish timeout
        _model_headers_to_kafka: Method to convert headers to Kafka format
    """

    # Type hints for attributes expected from parent class
    _config: ModelKafkaEventBusConfig
    _environment: str
    _group: str
    _producer: AIOKafkaProducer | None
    _producer_lock: asyncio.Lock
    _timeout_seconds: int

    def _init_dlq(self) -> None:
        """Initialize DLQ mixin state.

        Must be called during __init__ of the parent class.
        """
        # DLQ metrics tracking (copy-on-write pattern)
        self._dlq_metrics = ModelDlqMetrics.create_empty()
        self._dlq_metrics_lock = asyncio.Lock()

        # DLQ callback hooks for custom alerting integration
        self._dlq_callbacks: list[DlqCallbackType] = []
        self._dlq_callbacks_lock = asyncio.Lock()

    @property
    def dlq_metrics(self) -> ModelDlqMetrics:
        """Get a deep copy of the current DLQ metrics.

        Returns a deep copy of the metrics to prevent unintended mutation
        from external code. Deep copy ensures nested dicts (topic_counts,
        error_type_counts) are also copied. Thread-safe access to metrics data.

        Returns:
            Deep copy of the current DLQ metrics
        """
        return self._dlq_metrics.model_copy(deep=True)

    async def register_dlq_callback(
        self,
        callback: DlqCallbackType,
    ) -> Callable[[], Awaitable[None]]:
        """Register a callback to be invoked when messages are sent to DLQ.

        Callbacks receive a ModelDlqEvent containing comprehensive context
        about the DLQ operation, including success/failure status, original
        topic, error information, and correlation ID for tracing.

        Callbacks are invoked asynchronously after DLQ publish completes.
        If a callback raises an exception, it is logged but does not affect
        other callbacks or the DLQ publish operation.

        Args:
            callback: Async function that receives ModelDlqEvent

        Returns:
            Async function to unregister the callback

        Example:
            ```python
            async def alert_on_dlq(event: ModelDlqEvent) -> None:
                if event.is_critical:
                    await pagerduty.trigger(
                        summary=f"DLQ publish failed: {event.dlq_error_type}",
                        severity="critical",
                    )
                else:
                    logger.warning("Message sent to DLQ", extra=event.to_log_context())

            unregister = await bus.register_dlq_callback(alert_on_dlq)
            # Later, to unregister:
            await unregister()
            ```
        """
        async with self._dlq_callbacks_lock:
            self._dlq_callbacks.append(callback)

        async def unregister() -> None:
            async with self._dlq_callbacks_lock:
                if callback in self._dlq_callbacks:
                    self._dlq_callbacks.remove(callback)

        return unregister

    async def _publish_to_dlq(
        self: ProtocolKafkaDlqHost,
        original_topic: str,
        failed_message: ModelEventMessage,
        error: Exception,
        correlation_id: UUID,
        *,
        consumer_group: str,
    ) -> None:
        """Publish failed message to dead letter queue with metrics and alerting.

        This method publishes messages that failed processing to the configured
        dead letter queue topic with comprehensive failure metadata for later
        analysis and retry. If DLQ publishing fails, the error is logged but
        does not crash the consumer.

        Features:
            - Structured logging with appropriate log levels (WARNING/ERROR)
            - Metrics tracking (dlq.messages.published, dlq.messages.failed)
            - Callback hooks for custom alerting integration
            - Correlation ID included in all log entries for tracing

        Args:
            original_topic: Original topic where message was consumed from.
                Must be a non-empty, non-whitespace string.
            failed_message: The message that failed processing
            error: The exception that caused the failure
            correlation_id: Correlation ID for tracking
            consumer_group: Consumer group ID that processed the message.
                Required for DLQ traceability.

        Note:
            This method logs errors if DLQ publishing fails but does not raise
            exceptions to prevent cascading failures in the consumer loop.
        """
        # Validate original_topic - reject whitespace-only values
        if not original_topic or not original_topic.strip():
            logger.error(
                "DLQ publish rejected: original_topic is empty or whitespace-only",
                extra={
                    "correlation_id": str(correlation_id),
                    "error_type": type(error).__name__,
                },
            )
            return

        # Track timing for metrics
        start_time = datetime.now(UTC)
        error_type = type(error).__name__

        # Get DLQ topic using convention-based resolution
        # This supports both explicit dead_letter_topic config and automatic
        # topic generation following ONEX conventions: <env>.dlq.<category>.v1
        dlq_topic = self._config.get_dlq_topic()

        # Sanitize error message to prevent credential leakage in DLQ
        sanitized_failure_reason = sanitize_error_message(error)

        # Defensive decode for key and value to handle edge cases
        # This matches the pattern in _publish_raw_to_dlq for consistency
        try:
            key_str = (
                failed_message.key.decode("utf-8", errors="replace")
                if failed_message.key
                else None
            )
        except Exception:
            key_str = "<decode_failed>"

        try:
            value_str = (
                failed_message.value.decode("utf-8", errors="replace")
                if failed_message.value
                else "<no_value>"
            )
        except Exception:
            value_str = "<decode_failed>"

        # Build DLQ message with failure metadata
        dlq_payload = {
            "original_topic": original_topic,
            "original_message": {
                "key": key_str,
                "value": value_str,
                "offset": failed_message.offset,
                "partition": failed_message.partition,
            },
            "failure_reason": sanitized_failure_reason,
            "failure_timestamp": start_time.isoformat(),
            "correlation_id": str(correlation_id),
            "retry_count": failed_message.headers.retry_count,
            "error_type": error_type,
        }

        # Create DLQ headers with failure metadata
        dlq_headers = ModelEventHeaders(
            source=self._environment,
            event_type="dlq_message",
            content_type="application/json",
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC),
        )

        # Convert DLQ payload to JSON bytes with explicit error handling
        # for non-serializable content
        try:
            dlq_value = json.dumps(dlq_payload).encode("utf-8")
        except (TypeError, ValueError) as json_error:
            # Handle non-serializable envelope content by falling back to repr
            logger.warning(
                "DLQ payload contains non-serializable data, using repr fallback",
                extra={
                    "correlation_id": str(correlation_id),
                    "original_topic": original_topic,
                    "json_error": str(json_error),
                },
            )
            # Create a safe fallback payload with repr of original values
            fallback_payload = {
                "original_topic": original_topic,
                "original_message": {
                    "key": repr(key_str),
                    "value": "<non-serializable>",
                    "offset": failed_message.offset,
                    "partition": failed_message.partition,
                },
                "failure_reason": sanitized_failure_reason,
                "failure_timestamp": start_time.isoformat(),
                "correlation_id": str(correlation_id),
                "retry_count": failed_message.headers.retry_count,
                "error_type": error_type,
                "serialization_fallback": True,
            }
            dlq_value = json.dumps(fallback_payload).encode("utf-8")

        # Variables for event creation
        success = False
        dlq_error_type: str | None = None
        dlq_error_message: str | None = None

        # Publish to DLQ (without retry - best effort)
        # Capture producer reference and headers under lock, then send outside lock
        producer = None
        kafka_headers: list[tuple[str, bytes]] | None = None
        try:
            async with self._producer_lock:
                if self._producer is None:
                    dlq_error_type = "ProducerUnavailable"
                    dlq_error_message = "Producer not initialized or closed"
                    logger.error(
                        "DLQ publish failed: producer not available",
                        extra={
                            "original_topic": original_topic,
                            "dlq_topic": dlq_topic,
                            "correlation_id": str(correlation_id),
                            "error_type": error_type,
                            "retry_count": failed_message.headers.retry_count,
                        },
                    )
                else:
                    kafka_headers = self._model_headers_to_kafka(dlq_headers)
                    kafka_headers.extend(
                        [
                            ("original_topic", original_topic.encode("utf-8")),
                            (
                                "failure_reason",
                                sanitized_failure_reason.encode("utf-8"),
                            ),
                            (
                                "failure_timestamp",
                                start_time.isoformat().encode("utf-8"),
                            ),
                        ]
                    )
                    producer = self._producer

            # Send and wait for completion with timeout (outside producer lock)
            # Using send_and_wait() wrapped in wait_for() for cleaner timeout handling
            if producer is not None and kafka_headers is not None:
                await asyncio.wait_for(
                    producer.send_and_wait(
                        dlq_topic,
                        value=dlq_value,
                        key=failed_message.key,
                        headers=kafka_headers,
                    ),
                    timeout=self._timeout_seconds,
                )
                success = True

        except Exception as dlq_error:
            dlq_error_type = type(dlq_error).__name__
            dlq_error_message = sanitize_error_message(dlq_error)
            logger.exception(
                "DLQ publish failed: message may be lost",
                extra={
                    "original_topic": original_topic,
                    "dlq_topic": dlq_topic,
                    "correlation_id": str(correlation_id),
                    "error_type": error_type,
                    "dlq_error_type": dlq_error_type,
                    "dlq_error_message": dlq_error_message,
                    "retry_count": failed_message.headers.retry_count,
                    "message_offset": failed_message.offset,
                    "message_partition": failed_message.partition,
                },
            )

        # Calculate duration for metrics
        end_time = datetime.now(UTC)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Log successful DLQ publish at WARNING level
        if success:
            logger.warning(
                "Message published to DLQ due to processing failure",
                extra={
                    "original_topic": original_topic,
                    "dlq_topic": dlq_topic,
                    "correlation_id": str(correlation_id),
                    "error_type": error_type,
                    "error_message": sanitized_failure_reason,
                    "retry_count": failed_message.headers.retry_count,
                    "message_offset": failed_message.offset,
                    "message_partition": failed_message.partition,
                    "duration_ms": round(duration_ms, 2),
                },
            )

        # Create DLQ event for metrics and callbacks
        # Convert message_offset to string for type consistency with ModelDlqEvent
        # (which expects str | None) and consistency with _publish_raw_to_dlq
        message_offset_str = (
            str(failed_message.offset) if failed_message.offset is not None else None
        )
        dlq_event = ModelDlqEvent(
            original_topic=original_topic,
            dlq_topic=dlq_topic,
            correlation_id=correlation_id,
            error_type=error_type,
            error_message=sanitized_failure_reason,
            retry_count=failed_message.headers.retry_count,
            message_offset=message_offset_str,
            message_partition=failed_message.partition,
            success=success,
            dlq_error_type=dlq_error_type,
            dlq_error_message=dlq_error_message,
            timestamp=end_time,
            environment=self._environment,
            consumer_group=consumer_group,
        )

        # Update DLQ metrics (copy-on-write pattern)
        async with self._dlq_metrics_lock:
            self._dlq_metrics = self._dlq_metrics.record_dlq_publish(
                original_topic=original_topic,
                error_type=error_type,
                success=success,
                duration_ms=duration_ms,
            )

        # Invoke DLQ callbacks for custom alerting
        await self._invoke_dlq_callbacks(dlq_event)

    async def _invoke_dlq_callbacks(self, event: ModelDlqEvent) -> None:
        """Invoke registered DLQ callbacks with error isolation.

        Callbacks are invoked sequentially. If a callback raises an exception,
        it is logged but does not prevent other callbacks from executing.

        Args:
            event: The DLQ event to pass to callbacks
        """
        # Get a copy of callbacks under lock
        async with self._dlq_callbacks_lock:
            callbacks = list(self._dlq_callbacks)

        for callback in callbacks:
            try:
                await callback(event)
            except Exception as callback_error:
                logger.warning(
                    "DLQ callback raised exception",
                    extra={
                        "callback": getattr(callback, "__name__", str(callback)),
                        "correlation_id": str(event.correlation_id),
                        "original_topic": event.original_topic,
                        "callback_error_type": type(callback_error).__name__,
                        "callback_error_message": sanitize_error_message(
                            callback_error
                        ),
                    },
                    exc_info=True,
                )

    async def _publish_raw_to_dlq(
        self: ProtocolKafkaDlqHost,
        original_topic: str,
        raw_msg: object,
        error: Exception,
        correlation_id: UUID,
        failure_type: str,
        *,
        consumer_group: str,
    ) -> None:
        """Publish raw Kafka message to DLQ when deserialization fails.

        This method handles cases where message conversion fails before we have
        a ModelEventMessage. It extracts raw data directly from the Kafka
        ConsumerRecord for DLQ payload construction.

        Args:
            original_topic: Original topic where message was consumed from.
                Must be a non-empty, non-whitespace string.
            raw_msg: Raw Kafka ConsumerRecord that failed conversion
            error: The exception that caused the failure
            correlation_id: Correlation ID for tracking
            failure_type: Type of failure (e.g., "deserialization_error")
            consumer_group: Consumer group ID that processed the message.
                Required for DLQ traceability.

        Note:
            This method logs errors if DLQ publishing fails but does not raise
            exceptions to prevent cascading failures in the consumer loop.
        """
        # Validate original_topic - reject whitespace-only values
        if not original_topic or not original_topic.strip():
            logger.error(
                "DLQ publish rejected: original_topic is empty or whitespace-only",
                extra={
                    "correlation_id": str(correlation_id),
                    "error_type": type(error).__name__,
                    "failure_type": failure_type,
                },
            )
            return

        # Track timing for metrics
        start_time = datetime.now(UTC)
        error_type = type(error).__name__

        # Get DLQ topic using convention-based resolution
        # This supports both explicit dead_letter_topic config and automatic
        # topic generation following ONEX conventions: <env>.dlq.<category>.v1
        dlq_topic = self._config.get_dlq_topic()

        # Sanitize error message
        sanitized_failure_reason = sanitize_error_message(error)

        # Extract raw data from Kafka message
        raw_key = getattr(raw_msg, "key", None)
        raw_value = getattr(raw_msg, "value", b"")
        raw_offset = getattr(raw_msg, "offset", None)
        raw_partition = getattr(raw_msg, "partition", None)

        # Safe decode with error replacement
        try:
            key_str = (
                raw_key.decode("utf-8", errors="replace")
                if isinstance(raw_key, bytes)
                else str(raw_key)
                if raw_key is not None
                else None
            )
        except Exception:
            key_str = "<decode_failed>"

        try:
            value_str = (
                raw_value.decode("utf-8", errors="replace")
                if isinstance(raw_value, bytes)
                else str(raw_value)
            )
        except Exception:
            value_str = "<decode_failed>"

        # Build DLQ message with failure metadata
        dlq_payload = {
            "original_topic": original_topic,
            "original_message": {
                "key": key_str,
                "value": value_str,
                "offset": raw_offset,
                "partition": raw_partition,
            },
            "failure_reason": sanitized_failure_reason,
            "failure_type": failure_type,
            "failure_timestamp": start_time.isoformat(),
            "correlation_id": str(correlation_id),
            "retry_count": 0,
            "error_type": error_type,
        }

        # Create DLQ headers
        dlq_headers = ModelEventHeaders(
            source=self._environment,
            event_type="dlq_raw_message",
            content_type="application/json",
            correlation_id=correlation_id,
            timestamp=start_time,
        )

        # Convert DLQ payload to JSON bytes with explicit error handling
        # for non-serializable content
        try:
            dlq_value = json.dumps(dlq_payload).encode("utf-8")
        except (TypeError, ValueError) as json_error:
            # Handle non-serializable envelope content by falling back to repr
            logger.warning(
                "DLQ raw payload contains non-serializable data, using repr fallback",
                extra={
                    "correlation_id": str(correlation_id),
                    "original_topic": original_topic,
                    "failure_type": failure_type,
                    "json_error": str(json_error),
                },
            )
            # Create a safe fallback payload with repr of original values
            fallback_payload = {
                "original_topic": original_topic,
                "original_message": {
                    "key": repr(key_str),
                    "value": "<non-serializable>",
                    "offset": raw_offset,
                    "partition": raw_partition,
                },
                "failure_reason": sanitized_failure_reason,
                "failure_type": failure_type,
                "failure_timestamp": start_time.isoformat(),
                "correlation_id": str(correlation_id),
                "retry_count": 0,
                "error_type": error_type,
                "serialization_fallback": True,
            }
            dlq_value = json.dumps(fallback_payload).encode("utf-8")
        dlq_key = raw_key if isinstance(raw_key, bytes) else None

        # Variables for event creation
        success = False
        dlq_error_type: str | None = None
        dlq_error_message: str | None = None

        # Publish to DLQ
        # Capture producer reference and headers under lock, then send outside lock
        producer = None
        kafka_headers: list[tuple[str, bytes]] | None = None
        try:
            async with self._producer_lock:
                if self._producer is None:
                    dlq_error_type = "ProducerUnavailable"
                    dlq_error_message = "Producer not initialized or closed"
                    logger.error(
                        "DLQ publish failed: producer not available for raw message",
                        extra={
                            "original_topic": original_topic,
                            "dlq_topic": dlq_topic,
                            "correlation_id": str(correlation_id),
                            "error_type": error_type,
                            "failure_type": failure_type,
                        },
                    )
                else:
                    kafka_headers = self._model_headers_to_kafka(dlq_headers)
                    kafka_headers.extend(
                        [
                            ("original_topic", original_topic.encode("utf-8")),
                            ("failure_type", failure_type.encode("utf-8")),
                            (
                                "failure_reason",
                                sanitized_failure_reason.encode("utf-8"),
                            ),
                            (
                                "failure_timestamp",
                                start_time.isoformat().encode("utf-8"),
                            ),
                        ]
                    )
                    producer = self._producer

            # Send and wait for completion with timeout (outside producer lock)
            # Using send_and_wait() wrapped in wait_for() for cleaner timeout handling
            if producer is not None and kafka_headers is not None:
                await asyncio.wait_for(
                    producer.send_and_wait(
                        dlq_topic,
                        value=dlq_value,
                        key=dlq_key,
                        headers=kafka_headers,
                    ),
                    timeout=self._timeout_seconds,
                )
                success = True

        except Exception as dlq_error:
            dlq_error_type = type(dlq_error).__name__
            dlq_error_message = sanitize_error_message(dlq_error)
            logger.exception(
                "DLQ publish failed for raw message: message may be lost",
                extra={
                    "original_topic": original_topic,
                    "dlq_topic": dlq_topic,
                    "correlation_id": str(correlation_id),
                    "error_type": error_type,
                    "failure_type": failure_type,
                    "dlq_error_type": dlq_error_type,
                    "dlq_error_message": dlq_error_message,
                    "message_offset": raw_offset,
                    "message_partition": raw_partition,
                },
            )

        # Calculate duration for metrics
        end_time = datetime.now(UTC)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Log successful DLQ publish
        if success:
            logger.warning(
                "Raw message published to DLQ due to deserialization/validation failure",
                extra={
                    "original_topic": original_topic,
                    "dlq_topic": dlq_topic,
                    "correlation_id": str(correlation_id),
                    "error_type": error_type,
                    "error_message": sanitized_failure_reason,
                    "failure_type": failure_type,
                    "message_offset": raw_offset,
                    "message_partition": raw_partition,
                    "duration_ms": round(duration_ms, 2),
                },
            )

        # Create DLQ event for metrics and callbacks
        message_offset_str = str(raw_offset) if raw_offset is not None else None
        dlq_event = ModelDlqEvent(
            original_topic=original_topic,
            dlq_topic=dlq_topic,
            correlation_id=correlation_id,
            error_type=error_type,
            error_message=sanitized_failure_reason,
            retry_count=0,
            message_offset=message_offset_str,
            message_partition=raw_partition,
            success=success,
            dlq_error_type=dlq_error_type,
            dlq_error_message=dlq_error_message,
            timestamp=end_time,
            environment=self._environment,
            consumer_group=consumer_group,
        )

        # Update DLQ metrics
        async with self._dlq_metrics_lock:
            self._dlq_metrics = self._dlq_metrics.record_dlq_publish(
                original_topic=original_topic,
                error_type=error_type,
                success=success,
                duration_ms=duration_ms,
            )

        # Invoke DLQ callbacks
        await self._invoke_dlq_callbacks(dlq_event)


__all__: list[str] = ["MixinKafkaDlq", "DlqCallbackType"]
