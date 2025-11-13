"""Event models for metadata stamping service based on omnibase_3 schemas.

This module implements OnexEnvelopeV1 event format and specific event types
for metadata stamping operations.
"""

import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_serializer


class MetadataEventType(str, Enum):
    """Event types for metadata stamping operations."""

    STAMP_CREATED = "metadata.stamp.created"
    STAMP_VALIDATED = "metadata.stamp.validated"
    BATCH_PROCESSED = "metadata.batch.processed"


class SecurityContext(BaseModel):
    """Security context for events with optional HMAC-SHA256 signing."""

    actor_id: str = Field(..., description="ID of the actor performing the operation")
    session_id: Optional[str] = Field(None, description="Session identifier")
    security_level: str = Field(
        default="standard", description="Security level (standard, elevated, critical)"
    )
    hmac_signature: Optional[str] = Field(
        None, description="HMAC-SHA256 signature of the payload"
    )
    signing_key_id: Optional[str] = Field(
        None, description="ID of the key used for signing"
    )

    def sign_payload(
        self, payload: dict, secret_key: str, key_id: Optional[str] = None
    ) -> None:
        """Sign the payload with HMAC-SHA256.

        Args:
            payload: Event payload to sign
            secret_key: Secret key for signing
            key_id: Optional key identifier
        """
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        signature = hmac.new(
            secret_key.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        self.hmac_signature = signature
        if key_id:
            self.signing_key_id = key_id

    def verify_payload(self, payload: dict, secret_key: str) -> bool:
        """Verify the payload signature.

        Args:
            payload: Event payload to verify
            secret_key: Secret key for verification

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.hmac_signature:
            return False

        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected_signature = hmac.new(
            secret_key.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(self.hmac_signature, expected_signature)


class OnexEnvelopeV1(BaseModel):
    """OnexEnvelopeV1 event envelope based on omnibase_3 event schemas."""

    # Envelope metadata
    envelope_version: str = Field(default="1.0", description="Envelope format version")
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique event identifier"
    )
    event_type: MetadataEventType = Field(..., description="Type of metadata event")
    event_version: str = Field(default="1.0", description="Event schema version")

    # Timing and correlation
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Event timestamp"
    )
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    causation_id: Optional[str] = Field(None, description="Causation event ID")

    # Source information
    source_service: str = Field(
        default="metadata-stamping-service", description="Source service name"
    )
    source_version: str = Field(default="0.1.0", description="Source service version")
    source_instance: Optional[str] = Field(
        None, description="Source service instance ID"
    )
    source_node: Optional[str] = Field(None, description="Source node identifier")

    # Environment and routing
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )
    region: Optional[str] = Field(None, description="Geographic region")
    partition_key: Optional[str] = Field(None, description="Kafka partition key")

    # Security context
    security_context: Optional[SecurityContext] = Field(
        None, description="Security context"
    )

    # Event payload
    payload: dict[str, Any] = Field(..., description="Event-specific payload data")

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional event metadata"
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize timestamp to ISO format."""
        return value.isoformat()

    def to_kafka_topic(self) -> str:
        """Generate Kafka topic name based on event type and environment."""
        env_prefix = "dev" if self.environment == "development" else self.environment
        event_suffix = self.event_type.value.replace(".", "-")
        return f"{env_prefix}.omninode-bridge.metadata.{event_suffix}.v1"

    def get_kafka_key(self) -> str:
        """Generate Kafka message key for partitioning."""
        if self.partition_key:
            return self.partition_key
        if self.correlation_id:
            return self.correlation_id
        return self.event_id

    def add_security_context(
        self,
        actor_id: str,
        session_id: Optional[str] = None,
        security_level: str = "standard",
    ) -> SecurityContext:
        """Add security context to the event.

        Args:
            actor_id: ID of the actor
            session_id: Optional session ID
            security_level: Security level

        Returns:
            Created security context
        """
        self.security_context = SecurityContext(
            actor_id=actor_id, session_id=session_id, security_level=security_level
        )
        return self.security_context

    def sign_event(self, secret_key: str, key_id: Optional[str] = None) -> None:
        """Sign the event payload with HMAC-SHA256.

        Args:
            secret_key: Secret key for signing
            key_id: Optional key identifier
        """
        if not self.security_context:
            raise ValueError("Security context must be set before signing")

        self.security_context.sign_payload(self.payload, secret_key, key_id)


class MetadataStampCreatedEvent(OnexEnvelopeV1):
    """Event for metadata stamp creation."""

    event_type: MetadataEventType = Field(default=MetadataEventType.STAMP_CREATED)

    def __init__(
        self,
        stamp_id: str,
        file_hash: str,
        file_path: str,
        file_size: int,
        execution_time_ms: float,
        **kwargs,
    ):
        """Initialize stamp created event.

        Args:
            stamp_id: Unique stamp identifier
            file_hash: BLAKE3 hash of the file
            file_path: Path to the stamped file
            file_size: Size of the file in bytes
            execution_time_ms: Time taken to create the stamp
            **kwargs: Additional envelope parameters
        """
        payload = {
            "stamp_id": stamp_id,
            "file_hash": file_hash,
            "file_path": file_path,
            "file_size_bytes": file_size,
            "execution_time_ms": execution_time_ms,
            "hash_algorithm": "BLAKE3",
            "created_at": datetime.now(UTC).isoformat(),
        }

        super().__init__(payload=payload, **kwargs)

        # Use file_hash as partition key for related events
        self.partition_key = file_hash


class MetadataStampValidatedEvent(OnexEnvelopeV1):
    """Event for metadata stamp validation."""

    event_type: MetadataEventType = Field(default=MetadataEventType.STAMP_VALIDATED)

    def __init__(
        self,
        file_hash: str,
        validation_result: bool,
        validation_time_ms: float,
        error_message: Optional[str] = None,
        **kwargs,
    ):
        """Initialize stamp validated event.

        Args:
            file_hash: BLAKE3 hash that was validated
            validation_result: Whether validation succeeded
            validation_time_ms: Time taken to validate
            error_message: Optional error message if validation failed
            **kwargs: Additional envelope parameters
        """
        payload = {
            "file_hash": file_hash,
            "validation_result": validation_result,
            "validation_time_ms": validation_time_ms,
            "hash_algorithm": "BLAKE3",
            "validated_at": datetime.now(UTC).isoformat(),
        }

        if error_message:
            payload["error_message"] = error_message

        super().__init__(payload=payload, **kwargs)

        # Use file_hash as partition key for related events
        self.partition_key = file_hash


class MetadataBatchProcessedEvent(OnexEnvelopeV1):
    """Event for batch processing operations."""

    event_type: MetadataEventType = Field(default=MetadataEventType.BATCH_PROCESSED)

    def __init__(
        self,
        batch_id: str,
        total_files: int,
        successful_files: int,
        failed_files: int,
        total_processing_time_ms: float,
        **kwargs,
    ):
        """Initialize batch processed event.

        Args:
            batch_id: Unique batch identifier
            total_files: Total number of files in batch
            successful_files: Number of successfully processed files
            failed_files: Number of failed files
            total_processing_time_ms: Total processing time
            **kwargs: Additional envelope parameters
        """
        payload = {
            "batch_id": batch_id,
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": successful_files / total_files if total_files > 0 else 0.0,
            "total_processing_time_ms": total_processing_time_ms,
            "average_processing_time_ms": (
                total_processing_time_ms / total_files if total_files > 0 else 0.0
            ),
            "processed_at": datetime.now(UTC).isoformat(),
        }

        super().__init__(payload=payload, **kwargs)

        # Use batch_id as partition key for batch events
        self.partition_key = batch_id
