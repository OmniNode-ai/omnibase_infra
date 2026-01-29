# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test adapter implementing ProtocolEventPublisher bridged to EventBusInmemory.

This adapter provides a production-equivalent envelope format for testing,
eliminating the need for per-handler test adapters and ensuring test/production
parity for event publishing.

Key Features:
    - Implements full ProtocolEventPublisher interface from omnibase_spi
    - Serializes events using canonical ModelEventEnvelope format
    - Preserves correlation_id, causation_id, and metadata in envelope
    - Handles topic routing (default via event_type, or explicit override)
    - Encodes partition_key to bytes using canonical UTF-8 encoding

Usage:
    ```python
    from omnibase_infra.event_bus import EventBusInmemory
    from omnibase_infra.event_bus.testing import (
        AdapterProtocolEventPublisherInmemory,
        decode_inmemory_event,
    )

    bus = EventBusInmemory(environment="test", group="test-group")
    await bus.start()

    adapter = AdapterProtocolEventPublisherInmemory(bus)

    # Publish an event
    success = await adapter.publish(
        event_type="omninode.user.event.created.v1",
        payload={"user_id": "usr-123", "email": "user@example.com"},
        correlation_id="corr-456",
    )

    # Retrieve and decode for assertions
    history = await bus.get_event_history(limit=1)
    envelope = decode_inmemory_event(history[0].value)
    assert envelope.correlation_id is not None
    ```

References:
    - ProtocolEventPublisher: omnibase_spi.protocols.event_bus.protocol_event_publisher
    - ModelEventEnvelope: omnibase_core.models.events.model_event_envelope
    - Parent ticket: OMN-1611
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.types import JsonType
from omnibase_infra.event_bus.testing.model_publisher_metrics import (
    ModelPublisherMetrics,
)

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.types.typed_dict import TypedDictEnvelopeBuildParams
    from omnibase_spi.protocols.types.protocol_core_types import ContextValue

logger = logging.getLogger(__name__)


class AdapterProtocolEventPublisherInmemory:
    """Test adapter implementing ProtocolEventPublisher bridged to EventBusInmemory.

    This adapter provides production-equivalent envelope serialization for testing,
    ensuring that test events use the exact same format as production events.
    It eliminates the need for per-handler test adapters.

    Key Design Decisions:
        - Uses ModelEventEnvelope for canonical envelope format
        - Preserves all correlation tracking (correlation_id, causation_id)
        - Stores causation_id in metadata.tags since ModelEventEnvelope doesn't have a
          dedicated field for it
        - Topic routing: explicit topic parameter takes precedence over event_type
        - partition_key is encoded to UTF-8 bytes as per SPI specification

    Attributes:
        bus: The underlying EventBusInmemory instance.
        service_name: Service identifier included in envelope metadata.
        instance_id: Instance identifier for envelope source tracking.

    Example:
        ```python
        bus = EventBusInmemory()
        await bus.start()

        adapter = AdapterProtocolEventPublisherInmemory(
            bus=bus,
            service_name="test-service",
            instance_id="instance-001",
        )

        success = await adapter.publish(
            event_type="user.created",
            payload={"id": "123"},
            correlation_id="corr-abc",
        )
        assert success is True
        ```
    """

    def __init__(
        self,
        bus: EventBusInmemory,
        service_name: str = "test-service",
        instance_id: str | None = None,
    ) -> None:
        """Initialize the adapter with an EventBusInmemory instance.

        Args:
            bus: The EventBusInmemory instance to bridge to.
            service_name: Service name for envelope metadata. Defaults to "test-service".
            instance_id: Optional instance identifier. Defaults to a generated UUID.
        """
        self._bus = bus
        self._service_name = service_name
        self._instance_id = instance_id or str(uuid4())
        self._metrics = ModelPublisherMetrics()
        self._closed = False

    async def publish(
        self,
        event_type: str,
        payload: JsonType,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        metadata: dict[str, ContextValue] | None = None,
        topic: str | None = None,
        partition_key: str | None = None,
    ) -> bool:
        """Publish event with canonical ModelEventEnvelope serialization.

        Builds a ModelEventEnvelope from the provided parameters, serializes to JSON,
        and publishes to the underlying EventBusInmemory.

        Topic Routing:
            1. If `topic` is provided, use it directly (explicit override).
            2. Otherwise, derive topic from `event_type` (default routing).

        Correlation Tracking:
            - correlation_id: Stored in envelope.correlation_id
            - causation_id: Stored in envelope.metadata.tags["causation_id"]
            - Both IDs are preserved through serialization for full traceability

        Args:
            event_type: Fully-qualified event type (e.g., "omninode.user.event.created.v1").
            payload: Event payload data (dict, list, or primitive JSON types).
            correlation_id: Optional correlation ID for request tracing.
            causation_id: Optional causation ID for event sourcing chains.
            metadata: Optional additional metadata as context values.
            topic: Optional explicit topic override. When None, uses event_type as topic.
            partition_key: Optional partition key for message ordering.

        Returns:
            True if published successfully, False otherwise.

        Raises:
            RuntimeError: If adapter has been closed.
        """
        if self._closed:
            raise RuntimeError("Publisher has been closed")

        start_time = datetime.now(UTC)

        try:
            # Build envelope - parameters passed as dict to comply with ONEX parameter limit
            envelope = self._build_envelope(
                {
                    "event_type": event_type,
                    "payload": payload,
                    "correlation_id": correlation_id,
                    "causation_id": causation_id,
                    "metadata": metadata,
                }
            )

            # Determine target topic
            target_topic = topic if topic is not None else event_type

            # Encode partition key to bytes (UTF-8 canonical encoding)
            key_bytes: bytes | None = None
            if partition_key is not None:
                key_bytes = partition_key.encode("utf-8")

            # Serialize envelope to JSON bytes
            envelope_dict = envelope.model_dump(mode="json")
            value_bytes = json.dumps(envelope_dict).encode("utf-8")

            # Publish to underlying bus
            await self._bus.publish(
                topic=target_topic,
                key=key_bytes,
                value=value_bytes,
            )

            # Update metrics
            elapsed_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            self._metrics.events_published += 1
            self._metrics.total_publish_time_ms += elapsed_ms
            self._metrics.avg_publish_time_ms = (
                self._metrics.total_publish_time_ms / self._metrics.events_published
            )
            self._metrics.current_failures = 0

            logger.debug(
                "Event published successfully",
                extra={
                    "event_type": event_type,
                    "topic": target_topic,
                    "correlation_id": correlation_id,
                    "elapsed_ms": elapsed_ms,
                },
            )

            return True

        except Exception as e:
            # NOTE: Intentionally broad exception catch for test adapter.
            # Test adapters should gracefully handle all errors and return False
            # rather than propagate exceptions that would crash test harnesses.
            # This allows test assertions on publish failure (e.g., assert result is False).
            # Update failure metrics
            self._metrics.events_failed += 1
            self._metrics.current_failures += 1

            logger.exception(
                "Failed to publish event",
                extra={
                    "event_type": event_type,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )

            return False

    def _build_envelope(
        self,
        params: TypedDictEnvelopeBuildParams,
    ) -> ModelEventEnvelope[JsonType]:
        """Build a ModelEventEnvelope from publish parameters.

        Args:
            params: Dictionary containing:
                - event_type: The event type identifier
                - payload: The event payload
                - correlation_id: Optional correlation ID
                - causation_id: Optional causation ID
                - metadata: Optional additional metadata

        Returns:
            Configured ModelEventEnvelope ready for serialization.
        """
        event_type = str(params["event_type"])
        payload = cast("JsonType", params["payload"])
        correlation_id = params.get("correlation_id")
        causation_id = params.get("causation_id")
        metadata = params.get("metadata")

        # Convert correlation_id string to UUID if provided
        corr_uuid: UUID | None = None
        if correlation_id is not None:
            try:
                corr_uuid = UUID(str(correlation_id))
            except ValueError:
                # If not a valid UUID, generate one and log the original for debugging
                corr_uuid = uuid4()
                logger.warning(
                    "correlation_id is not a valid UUID, generating new UUID (original logged)",
                    extra={
                        "original_correlation_id": correlation_id,
                        "generated_uuid": str(corr_uuid),
                    },
                )

        # Build metadata tags
        tags: dict[str, str] = {
            "event_type": event_type,
            "service_name": self._service_name,
            "instance_id": self._instance_id,
        }

        # Store causation_id in tags (ModelEventEnvelope doesn't have dedicated field)
        if causation_id is not None:
            tags["causation_id"] = str(causation_id)

        # Merge additional metadata context values into tags
        if metadata is not None and isinstance(metadata, dict):
            for key, value in metadata.items():
                # Context values may have a serialize_for_context method or be simple types
                if hasattr(value, "serialize_for_context"):
                    serialized = value.serialize_for_context()
                    tags[key] = json.dumps(serialized)
                elif hasattr(value, "value"):
                    # ProtocolContext*Value types have a value attribute
                    tags[key] = str(value.value)
                else:
                    tags[key] = str(value)

        envelope_metadata = ModelEnvelopeMetadata(tags=tags)

        # Build the envelope
        envelope: ModelEventEnvelope[JsonType] = ModelEventEnvelope(
            payload=payload,
            correlation_id=corr_uuid,
            source_tool=f"{self._service_name}.{self._instance_id}",
            metadata=envelope_metadata,
        )

        return envelope

    async def get_metrics(self) -> JsonType:
        """Get publisher metrics.

        Returns:
            Dictionary with metrics including:
            - events_published: Total successful publishes
            - events_failed: Total failed publishes
            - events_sent_to_dlq: Always 0 for inmemory (no DLQ)
            - total_publish_time_ms: Cumulative publish time
            - avg_publish_time_ms: Average publish latency
            - circuit_breaker_opens: Always 0 for inmemory
            - retries_attempted: Always 0 for inmemory
            - circuit_breaker_status: Always "closed" for inmemory
            - current_failures: Current consecutive failure count
        """
        return self._metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset all publisher metrics to initial values.

        Useful for test isolation when reusing an adapter across multiple
        test cases without recreating the adapter instance.

        Note:
            This method does NOT affect the closed state of the adapter.
            If the adapter has been closed, it remains closed after reset.

        Example:
            ```python
            adapter = AdapterProtocolEventPublisherInmemory(bus)
            await adapter.publish(...)  # metrics.events_published = 1

            adapter.reset_metrics()  # metrics.events_published = 0
            await adapter.publish(...)  # metrics.events_published = 1
            ```
        """
        self._metrics = ModelPublisherMetrics()
        logger.debug(
            "Publisher metrics reset",
            extra={
                "service_name": self._service_name,
                "instance_id": self._instance_id,
            },
        )

    async def close(self, timeout_seconds: float = 30.0) -> None:
        """Close the publisher.

        For the inmemory adapter, this simply marks the adapter as closed.
        No actual cleanup is needed since EventBusInmemory handles its own lifecycle.

        Args:
            timeout_seconds: Timeout for cleanup (unused for inmemory).
        """
        self._closed = True
        logger.info(
            "AdapterProtocolEventPublisherInmemory closed",
            extra={
                "service_name": self._service_name,
                "instance_id": self._instance_id,
            },
        )


def decode_inmemory_event(value: bytes) -> ModelEventEnvelope[object]:
    """Decode an inmemory event bus message value to ModelEventEnvelope.

    This helper function decodes the JSON bytes from EventBusInmemory message
    values back into a ModelEventEnvelope for test assertions.

    Args:
        value: The bytes value from ModelEventMessage.value

    Returns:
        Decoded ModelEventEnvelope with payload typed as object.

    Raises:
        json.JSONDecodeError: If value is not valid JSON.
        pydantic.ValidationError: If decoded JSON doesn't match envelope schema.

    Example:
        ```python
        history = await bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.correlation_id is not None
        assert envelope.metadata.tags.get("event_type") == "user.created"
        assert envelope.payload["user_id"] == "usr-123"
        ```
    """
    decoded_dict = json.loads(value.decode("utf-8"))
    return ModelEventEnvelope[object].model_validate(decoded_dict)


__all__: list[str] = [
    "AdapterProtocolEventPublisherInmemory",
    "decode_inmemory_event",
]
