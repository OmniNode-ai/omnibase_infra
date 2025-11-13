#!/usr/bin/env python3
"""
OnexEnvelopeV1 - Standard event envelope for Kafka messaging.

Provides a consistent wrapper for all Kafka events with metadata,
versioning, and traceability support.

ONEX v2.0 Compliance:
- Model-based naming: ModelOnexEnvelopeV1
- Strong typing with Pydantic v2
- Correlation tracking and distributed tracing
"""

from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer


class ModelOnexEnvelopeV1(BaseModel):
    """
    Standard envelope for wrapping Kafka event payloads.

    Provides consistent metadata, versioning, and tracing across all events
    in the omninode ecosystem.

    Attributes:
        envelope_version: Version of envelope format (always "1.0")
        event_id: Unique identifier for this event instance
        event_type: Type of event (e.g., "NODE_INTROSPECTION", "REGISTRY_REQUEST")
        event_version: Event schema version
        event_timestamp: When the event was created
        source_node_id: Identifier of the node that generated this event
        source_service: Source service name (for compatibility)
        source_version: Source service version (for compatibility)
        source_instance: Source service instance ID (for compatibility)
        correlation_id: Optional correlation ID for tracking related events
        causation_id: Optional causation event ID
        environment: Environment (development, staging, production)
        region: Geographic region
        partition_key: Kafka partition key
        payload: The actual event data (ModelNodeIntrospectionEvent, etc.)
        metadata: Additional envelope metadata

    Usage:
        ```python
        envelope = ModelOnexEnvelopeV1(
            event_type="NODE_INTROSPECTION",
            source_node_id="metadata-stamping-1",
            payload={
                "node_id": "metadata-stamping-1",
                "node_type": "effect",
                "capabilities": {...},
                "endpoints": {...}
            }
        )
        ```
    """

    envelope_version: str = Field(
        default="1.0",
        description="Envelope format version",
    )

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event instance",
    )

    event_type: str = Field(
        ...,
        description="Type of event (e.g., NODE_INTROSPECTION, REGISTRY_REQUEST)",
        examples=[
            "NODE_INTROSPECTION",
            "REGISTRY_REQUEST_INTROSPECTION",
            "NODE_HEARTBEAT",
        ],
    )

    event_version: str = Field(
        default="1.0",
        description="Event schema version",
    )

    event_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the event was created",
    )

    # Source information
    source_node_id: str = Field(
        ...,
        description="Identifier of the node that generated this event",
    )

    source_service: Optional[str] = Field(
        default="omninode-bridge",
        description="Source service name",
    )

    source_version: Optional[str] = Field(
        default="1.0.0",
        description="Source service version",
    )

    source_instance: Optional[str] = Field(
        default=None,
        description="Source service instance ID",
    )

    # Correlation and tracing
    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Optional correlation ID for tracking related events",
    )

    causation_id: Optional[str] = Field(
        default=None,
        description="Causation event ID",
    )

    # Environment and routing
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )

    region: Optional[str] = Field(
        default=None,
        description="Geographic region",
    )

    partition_key: Optional[str] = Field(
        default=None,
        description="Kafka partition key",
    )

    # Payload and metadata
    payload: dict[str, Any] = Field(
        ...,
        description="The actual event data (deserialized event model)",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional envelope metadata",
    )

    @field_serializer("event_timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize timestamp to ISO format."""
        return value.isoformat()

    @field_serializer("event_id", "correlation_id")
    def serialize_uuid(self, value: Optional[UUID]) -> Optional[str]:
        """Serialize UUID to string."""
        return str(value) if value is not None else None

    @classmethod
    def create(
        cls,
        event_type: str,
        source_node_id: str,
        payload: dict[str, Any],
        correlation_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ModelOnexEnvelopeV1":
        """
        Factory method to create an envelope with defaults.

        Args:
            event_type: Type of event
            source_node_id: Source node identifier
            payload: Event payload data
            correlation_id: Optional correlation ID
            metadata: Optional metadata

        Returns:
            ModelOnexEnvelopeV1 instance
        """
        return cls(
            event_type=event_type,
            source_node_id=source_node_id,
            payload=payload,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert envelope to dictionary for Kafka publishing.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return self.model_dump(mode="json")

    def to_bytes(self) -> bytes:
        """
        Convert envelope to bytes for Kafka publishing.

        Returns:
            Bytes representation suitable for Kafka message value
        """
        import json

        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelOnexEnvelopeV1":
        """
        Create envelope from dictionary (deserialize from Kafka).

        Args:
            data: Dictionary from Kafka message

        Returns:
            ModelOnexEnvelopeV1 instance
        """
        return cls(**data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ModelOnexEnvelopeV1":
        """
        Create envelope from bytes (deserialize from Kafka message).

        Args:
            data: Bytes from Kafka message

        Returns:
            ModelOnexEnvelopeV1 instance
        """
        import json

        return cls.from_dict(json.loads(data.decode("utf-8")))

    def to_kafka_topic(self, service_prefix: str = "omninode-bridge") -> str:
        """
        Generate Kafka topic name based on event type and environment.

        Args:
            service_prefix: Service prefix for topic name (default: "omninode-bridge")

        Returns:
            Kafka topic name in format: {env}.{service}.{event_type}.v1
        """
        env_prefix = "dev" if self.environment == "development" else self.environment
        event_suffix = self.event_type.replace("_", "-").lower()
        return f"{env_prefix}.{service_prefix}.{event_suffix}.v1"

    def get_kafka_key(self) -> str:
        """
        Generate Kafka message key for partitioning.

        Returns:
            Kafka partition key (priority: partition_key > correlation_id > source_node_id > event_id)
        """
        if self.partition_key:
            return self.partition_key
        if self.correlation_id:
            return str(self.correlation_id)
        if self.source_node_id:
            return self.source_node_id
        return str(self.event_id)
