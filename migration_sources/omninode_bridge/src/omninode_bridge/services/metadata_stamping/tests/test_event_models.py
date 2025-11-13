"""Tests for metadata stamping event models."""

import json
from datetime import UTC, datetime

import pytest

from ..events.models import (
    MetadataBatchProcessedEvent,
    MetadataEventType,
    MetadataStampCreatedEvent,
    MetadataStampValidatedEvent,
    OnexEnvelopeV1,
    SecurityContext,
)


class TestSecurityContext:
    """Test security context functionality."""

    def test_security_context_creation(self):
        """Test creating a security context."""
        context = SecurityContext(
            actor_id="user123", session_id="session456", security_level="elevated"
        )

        assert context.actor_id == "user123"
        assert context.session_id == "session456"
        assert context.security_level == "elevated"
        assert context.hmac_signature is None
        assert context.signing_key_id is None

    def test_payload_signing(self):
        """Test HMAC-SHA256 payload signing."""
        context = SecurityContext(actor_id="user123")
        payload = {"file_hash": "abc123", "operation": "stamp_created"}
        secret_key = "test_secret_key"
        key_id = "key_001"

        context.sign_payload(payload, secret_key, key_id)

        assert context.hmac_signature is not None
        assert len(context.hmac_signature) == 64  # SHA256 hex digest length
        assert context.signing_key_id == key_id

    def test_payload_verification_success(self):
        """Test successful payload verification."""
        context = SecurityContext(actor_id="user123")
        payload = {"file_hash": "abc123", "operation": "stamp_created"}
        secret_key = "test_secret_key"

        context.sign_payload(payload, secret_key)
        assert context.verify_payload(payload, secret_key) is True

    def test_payload_verification_failure(self):
        """Test payload verification failure with wrong key."""
        context = SecurityContext(actor_id="user123")
        payload = {"file_hash": "abc123", "operation": "stamp_created"}
        secret_key = "test_secret_key"
        wrong_key = "wrong_secret_key"

        context.sign_payload(payload, secret_key)
        assert context.verify_payload(payload, wrong_key) is False

    def test_payload_verification_no_signature(self):
        """Test payload verification with no signature."""
        context = SecurityContext(actor_id="user123")
        payload = {"file_hash": "abc123", "operation": "stamp_created"}
        secret_key = "test_secret_key"

        assert context.verify_payload(payload, secret_key) is False


class TestOnexEnvelopeV1:
    """Test OnexEnvelopeV1 base event envelope."""

    def test_envelope_creation(self):
        """Test creating a basic event envelope."""
        payload = {"test": "data"}
        envelope = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload=payload
        )

        assert envelope.envelope_version == "1.0"
        assert envelope.event_type == MetadataEventType.STAMP_CREATED
        assert envelope.event_version == "1.0"
        assert envelope.source_service == "metadata-stamping-service"
        assert envelope.source_version == "0.1.0"
        assert envelope.environment == "development"
        assert envelope.payload == payload
        assert isinstance(envelope.timestamp, datetime)
        assert envelope.timestamp.tzinfo == UTC

    def test_envelope_serialization(self):
        """Test envelope serialization to dict."""
        payload = {"file_hash": "abc123"}
        envelope = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED,
            payload=payload,
            correlation_id="corr123",
        )

        data = envelope.model_dump()

        assert data["event_type"] == "metadata.stamp.created"
        assert data["payload"] == payload
        assert data["correlation_id"] == "corr123"
        assert isinstance(data["timestamp"], str)  # Should be serialized as ISO string

    def test_kafka_topic_generation(self):
        """Test Kafka topic name generation."""
        envelope = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        topic = envelope.to_kafka_topic()
        assert topic == "dev.omninode-bridge.metadata.metadata-stamp-created.v1"

        # Test production environment
        envelope.environment = "production"
        topic = envelope.to_kafka_topic()
        assert topic == "production.omninode-bridge.metadata.metadata-stamp-created.v1"

    def test_kafka_key_generation(self):
        """Test Kafka message key generation."""
        envelope = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        # Default: uses event_id
        key = envelope.get_kafka_key()
        assert key == envelope.event_id

        # With partition_key
        envelope.partition_key = "custom_key"
        key = envelope.get_kafka_key()
        assert key == "custom_key"

        # With correlation_id but no partition_key
        envelope.partition_key = None
        envelope.correlation_id = "corr123"
        key = envelope.get_kafka_key()
        assert key == "corr123"

    def test_security_context_addition(self):
        """Test adding security context to envelope."""
        envelope = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        context = envelope.add_security_context(
            actor_id="user123", session_id="session456", security_level="elevated"
        )

        assert envelope.security_context == context
        assert context.actor_id == "user123"
        assert context.session_id == "session456"
        assert context.security_level == "elevated"

    def test_event_signing(self):
        """Test event signing with security context."""
        envelope = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"file_hash": "abc123"}
        )

        envelope.add_security_context(actor_id="user123")
        envelope.sign_event("test_secret", "key001")

        assert envelope.security_context.hmac_signature is not None
        assert envelope.security_context.signing_key_id == "key001"

    def test_event_signing_without_security_context(self):
        """Test event signing fails without security context."""
        envelope = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"file_hash": "abc123"}
        )

        with pytest.raises(ValueError, match="Security context must be set"):
            envelope.sign_event("test_secret")


class TestMetadataStampCreatedEvent:
    """Test metadata stamp created event."""

    def test_stamp_created_event_creation(self):
        """Test creating a stamp created event."""
        event = MetadataStampCreatedEvent(
            stamp_id="stamp123",
            file_hash="abc123def456",
            file_path="/path/to/file.txt",
            file_size=1024,
            execution_time_ms=15.5,
            correlation_id="corr123",
        )

        assert event.event_type == MetadataEventType.STAMP_CREATED
        assert event.correlation_id == "corr123"
        assert event.partition_key == "abc123def456"  # Uses file_hash

        payload = event.payload
        assert payload["stamp_id"] == "stamp123"
        assert payload["file_hash"] == "abc123def456"
        assert payload["file_path"] == "/path/to/file.txt"
        assert payload["file_size_bytes"] == 1024
        assert payload["execution_time_ms"] == 15.5
        assert payload["hash_algorithm"] == "BLAKE3"
        assert "created_at" in payload

    def test_stamp_created_event_kafka_topic(self):
        """Test Kafka topic for stamp created event."""
        event = MetadataStampCreatedEvent(
            stamp_id="stamp123",
            file_hash="abc123",
            file_path="/test",
            file_size=100,
            execution_time_ms=10.0,
        )

        topic = event.to_kafka_topic()
        assert topic == "dev.omninode-bridge.metadata.metadata-stamp-created.v1"


class TestMetadataStampValidatedEvent:
    """Test metadata stamp validated event."""

    def test_stamp_validated_success_event(self):
        """Test creating a successful validation event."""
        event = MetadataStampValidatedEvent(
            file_hash="abc123def456",
            validation_result=True,
            validation_time_ms=8.2,
            correlation_id="corr123",
        )

        assert event.event_type == MetadataEventType.STAMP_VALIDATED
        assert event.correlation_id == "corr123"
        assert event.partition_key == "abc123def456"  # Uses file_hash

        payload = event.payload
        assert payload["file_hash"] == "abc123def456"
        assert payload["validation_result"] is True
        assert payload["validation_time_ms"] == 8.2
        assert payload["hash_algorithm"] == "BLAKE3"
        assert "validated_at" in payload
        assert "error_message" not in payload

    def test_stamp_validated_failure_event(self):
        """Test creating a failed validation event."""
        event = MetadataStampValidatedEvent(
            file_hash="abc123def456",
            validation_result=False,
            validation_time_ms=12.1,
            error_message="Hash mismatch",
            correlation_id="corr123",
        )

        payload = event.payload
        assert payload["validation_result"] is False
        assert payload["error_message"] == "Hash mismatch"

    def test_stamp_validated_event_kafka_topic(self):
        """Test Kafka topic for stamp validated event."""
        event = MetadataStampValidatedEvent(
            file_hash="abc123", validation_result=True, validation_time_ms=5.0
        )

        topic = event.to_kafka_topic()
        assert topic == "dev.omninode-bridge.metadata.metadata-stamp-validated.v1"


class TestMetadataBatchProcessedEvent:
    """Test metadata batch processed event."""

    def test_batch_processed_event_creation(self):
        """Test creating a batch processed event."""
        event = MetadataBatchProcessedEvent(
            batch_id="batch123",
            total_files=10,
            successful_files=8,
            failed_files=2,
            total_processing_time_ms=1500.0,
            correlation_id="corr123",
        )

        assert event.event_type == MetadataEventType.BATCH_PROCESSED
        assert event.correlation_id == "corr123"
        assert event.partition_key == "batch123"  # Uses batch_id

        payload = event.payload
        assert payload["batch_id"] == "batch123"
        assert payload["total_files"] == 10
        assert payload["successful_files"] == 8
        assert payload["failed_files"] == 2
        assert payload["success_rate"] == 0.8
        assert payload["total_processing_time_ms"] == 1500.0
        assert payload["average_processing_time_ms"] == 150.0
        assert "processed_at" in payload

    def test_batch_processed_event_zero_files(self):
        """Test batch processed event with zero files."""
        event = MetadataBatchProcessedEvent(
            batch_id="batch123",
            total_files=0,
            successful_files=0,
            failed_files=0,
            total_processing_time_ms=0.0,
        )

        payload = event.payload
        assert payload["success_rate"] == 0.0
        assert payload["average_processing_time_ms"] == 0.0

    def test_batch_processed_event_kafka_topic(self):
        """Test Kafka topic for batch processed event."""
        event = MetadataBatchProcessedEvent(
            batch_id="batch123",
            total_files=5,
            successful_files=5,
            failed_files=0,
            total_processing_time_ms=500.0,
        )

        topic = event.to_kafka_topic()
        assert topic == "dev.omninode-bridge.metadata.metadata-batch-processed.v1"


class TestEventSerialization:
    """Test event serialization and JSON compatibility."""

    def test_event_json_serialization(self):
        """Test that events can be serialized to JSON."""
        event = MetadataStampCreatedEvent(
            stamp_id="stamp123",
            file_hash="abc123",
            file_path="/test",
            file_size=100,
            execution_time_ms=10.0,
        )

        # Should serialize without errors
        json_data = json.dumps(event.model_dump())
        assert isinstance(json_data, str)

        # Should deserialize back
        data = json.loads(json_data)
        assert data["event_type"] == "metadata.stamp.created"
        assert data["payload"]["stamp_id"] == "stamp123"

    def test_event_with_security_context_serialization(self):
        """Test event with security context serialization."""
        event = MetadataStampCreatedEvent(
            stamp_id="stamp123",
            file_hash="abc123",
            file_path="/test",
            file_size=100,
            execution_time_ms=10.0,
        )

        event.add_security_context(actor_id="user123", session_id="session456")
        event.sign_event("secret_key", "key001")

        # Should serialize without errors
        json_data = json.dumps(event.model_dump())
        data = json.loads(json_data)

        assert data["security_context"]["actor_id"] == "user123"
        assert data["security_context"]["session_id"] == "session456"
        assert "hmac_signature" in data["security_context"]
        assert data["security_context"]["signing_key_id"] == "key001"
