"""Tests for metadata stamping event publisher."""

from unittest.mock import AsyncMock, patch

import pytest

from ..events.models import MetadataEventType, OnexEnvelopeV1
from ..events.publisher import EventPublisher


class TestEventPublisherInitialization:
    """Test event publisher initialization."""

    def test_publisher_init_disabled(self):
        """Test publisher initialization with events disabled."""
        publisher = EventPublisher(enable_events=False)

        assert publisher.enable_events is False
        assert publisher.secret_key is None
        assert publisher.events_published == 0
        assert publisher.events_failed == 0

    def test_publisher_init_enabled(self):
        """Test publisher initialization with events enabled."""
        publisher = EventPublisher(enable_events=True, secret_key="test_secret")

        assert publisher.enable_events is True
        assert publisher.secret_key == "test_secret"
        assert publisher.key_id == "default"

    @pytest.mark.asyncio
    async def test_publisher_initialize_disabled(self):
        """Test publisher initialization when events are disabled."""
        publisher = EventPublisher(enable_events=False)

        result = await publisher.initialize()

        assert result is True
        assert publisher._kafka_client is None

    @pytest.mark.asyncio
    async def test_publisher_initialize_with_kafka_client(self):
        """Test publisher initialization with provided Kafka client."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.connect.return_value = None
        mock_kafka_client.is_connected = True

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        result = await publisher.initialize()

        assert result is True
        assert publisher._kafka_client == mock_kafka_client
        mock_kafka_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_publisher_initialize_kafka_unavailable(self):
        """Test publisher initialization when Kafka is unavailable."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.connect.return_value = None
        mock_kafka_client.is_connected = False

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        result = await publisher.initialize()

        # Should still return True (graceful degradation)
        assert result is True
        assert publisher._kafka_client == mock_kafka_client

    @pytest.mark.asyncio
    async def test_publisher_initialize_kafka_error(self):
        """Test publisher initialization when Kafka connection fails."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.connect.side_effect = Exception("Connection failed")

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        result = await publisher.initialize()

        # Should still return True (graceful degradation)
        assert result is True
        assert publisher.last_publish_error == "Connection failed"

    @pytest.mark.asyncio
    async def test_publisher_cleanup(self):
        """Test publisher cleanup."""
        mock_kafka_client = AsyncMock()
        publisher = EventPublisher(kafka_client=mock_kafka_client)
        publisher._owned_kafka_client = True

        await publisher.cleanup()

        mock_kafka_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_publisher_cleanup_no_owned_client(self):
        """Test publisher cleanup when not owning the client."""
        mock_kafka_client = AsyncMock()
        publisher = EventPublisher(kafka_client=mock_kafka_client)
        publisher._owned_kafka_client = False

        await publisher.cleanup()

        # Should not disconnect if we don't own the client
        mock_kafka_client.disconnect.assert_not_called()


class TestEventPublishing:
    """Test event publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_event_disabled(self):
        """Test publishing when events are disabled."""
        publisher = EventPublisher(enable_events=False)
        event = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        result = await publisher.publish_event(event)

        assert result is True  # Should succeed (no-op)

    @pytest.mark.asyncio
    async def test_publish_event_success(self):
        """Test successful event publishing."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.return_value = True

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        event = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        result = await publisher.publish_event(event, actor_id="user123")

        assert result is True
        assert publisher.events_published == 1
        assert event.security_context is not None
        assert event.security_context.actor_id == "user123"

        # Verify Kafka call
        mock_kafka_client.publish_raw_event.assert_called_once()
        call_args = mock_kafka_client.publish_raw_event.call_args
        assert call_args[1]["topic"] == event.to_kafka_topic()
        assert call_args[1]["key"] == event.get_kafka_key()

    @pytest.mark.asyncio
    async def test_publish_event_with_signing(self):
        """Test event publishing with HMAC signing."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.return_value = True

        publisher = EventPublisher(
            kafka_client=mock_kafka_client, enable_events=True, secret_key="test_secret"
        )

        event = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        result = await publisher.publish_event(event, actor_id="user123")

        assert result is True
        assert event.security_context.hmac_signature is not None
        assert event.security_context.signing_key_id == "default"

    @pytest.mark.asyncio
    async def test_publish_event_kafka_failure(self):
        """Test event publishing when Kafka fails."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.return_value = False

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        event = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        result = await publisher.publish_event(event)

        assert result is False
        assert publisher.events_failed == 1
        assert publisher.last_publish_error == "Kafka publish failed"

    @pytest.mark.asyncio
    async def test_publish_event_kafka_unavailable(self):
        """Test event publishing when Kafka is unavailable."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = False

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        event = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        result = await publisher.publish_event(event)

        # Should return True (graceful degradation)
        assert result is True
        mock_kafka_client.publish_raw_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_event_exception(self):
        """Test event publishing when an exception occurs."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.side_effect = Exception("Kafka error")

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        event = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        result = await publisher.publish_event(event)

        assert result is False
        assert publisher.events_failed == 1
        assert "Kafka error" in publisher.last_publish_error


class TestSpecificEventPublishing:
    """Test specific event type publishing methods."""

    @pytest.mark.asyncio
    async def test_publish_stamp_created(self):
        """Test publishing stamp created event."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.return_value = True

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        result = await publisher.publish_stamp_created(
            stamp_id="stamp123",
            file_hash="abc123",
            file_path="/test/file.txt",
            file_size=1024,
            execution_time_ms=15.5,
            correlation_id="corr123",
            actor_id="user123",
        )

        assert result is True
        assert publisher.events_published == 1

        # Verify the event was published with correct data
        call_args = mock_kafka_client.publish_raw_event.call_args
        event_data = call_args[1]["data"]
        assert event_data["event_type"] == "metadata.stamp.created"
        assert event_data["payload"]["stamp_id"] == "stamp123"
        assert event_data["payload"]["file_hash"] == "abc123"
        assert event_data["correlation_id"] == "corr123"

    @pytest.mark.asyncio
    async def test_publish_stamp_validated_success(self):
        """Test publishing successful stamp validation event."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.return_value = True

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        result = await publisher.publish_stamp_validated(
            file_hash="abc123",
            validation_result=True,
            validation_time_ms=8.2,
            correlation_id="corr123",
        )

        assert result is True
        call_args = mock_kafka_client.publish_raw_event.call_args
        event_data = call_args[1]["data"]
        assert event_data["event_type"] == "metadata.stamp.validated"
        assert event_data["payload"]["validation_result"] is True
        assert "error_message" not in event_data["payload"]

    @pytest.mark.asyncio
    async def test_publish_stamp_validated_failure(self):
        """Test publishing failed stamp validation event."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.return_value = True

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        result = await publisher.publish_stamp_validated(
            file_hash="abc123",
            validation_result=False,
            validation_time_ms=12.1,
            error_message="Hash mismatch",
            correlation_id="corr123",
        )

        assert result is True
        call_args = mock_kafka_client.publish_raw_event.call_args
        event_data = call_args[1]["data"]
        assert event_data["payload"]["validation_result"] is False
        assert event_data["payload"]["error_message"] == "Hash mismatch"

    @pytest.mark.asyncio
    async def test_publish_batch_processed(self):
        """Test publishing batch processed event."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.return_value = True

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        result = await publisher.publish_batch_processed(
            batch_id="batch123",
            total_files=10,
            successful_files=8,
            failed_files=2,
            total_processing_time_ms=1500.0,
            correlation_id="corr123",
        )

        assert result is True
        call_args = mock_kafka_client.publish_raw_event.call_args
        event_data = call_args[1]["data"]
        assert event_data["event_type"] == "metadata.batch.processed"
        assert event_data["payload"]["batch_id"] == "batch123"
        assert event_data["payload"]["total_files"] == 10
        assert event_data["payload"]["success_rate"] == 0.8


class TestEventPublisherMetrics:
    """Test event publisher metrics functionality."""

    def test_get_metrics_disabled(self):
        """Test getting metrics when events are disabled."""
        publisher = EventPublisher(enable_events=False)

        metrics = publisher.get_metrics()

        assert metrics["events_enabled"] is False
        assert metrics["events_published"] == 0
        assert metrics["events_failed"] == 0
        assert metrics["last_publish_error"] is None
        assert metrics["kafka_connected"] is False

    def test_get_metrics_enabled(self):
        """Test getting metrics when events are enabled."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)
        publisher.events_published = 5
        publisher.events_failed = 2
        publisher.last_publish_error = "Test error"

        metrics = publisher.get_metrics()

        assert metrics["events_enabled"] is True
        assert metrics["events_published"] == 5
        assert metrics["events_failed"] == 2
        assert metrics["last_publish_error"] == "Test error"
        assert metrics["kafka_connected"] is True

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that metrics are properly tracked during publishing."""
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_raw_event.side_effect = [
            True,
            False,
            Exception("Error"),
        ]

        publisher = EventPublisher(kafka_client=mock_kafka_client, enable_events=True)

        event = OnexEnvelopeV1(
            event_type=MetadataEventType.STAMP_CREATED, payload={"test": "data"}
        )

        # Successful publish
        await publisher.publish_event(event)
        assert publisher.events_published == 1
        assert publisher.events_failed == 0

        # Failed publish
        await publisher.publish_event(event)
        assert publisher.events_published == 1
        assert publisher.events_failed == 1

        # Exception publish
        await publisher.publish_event(event)
        assert publisher.events_published == 1
        assert publisher.events_failed == 2
        assert "Error" in publisher.last_publish_error


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    @patch.dict(
        "os.environ",
        {
            "METADATA_STAMPING_ENABLE_EVENTS": "true",
            "METADATA_STAMPING_EVENT_SECRET_KEY": "env_secret",
            "METADATA_STAMPING_EVENT_KEY_ID": "env_key_id",
            "ENVIRONMENT": "production",
            "INSTANCE_ID": "test-instance-123",
        },
    )
    def test_environment_configuration(self):
        """Test configuration from environment variables."""
        publisher = EventPublisher()

        assert publisher.enable_events is True
        assert publisher.secret_key == "env_secret"
        assert publisher.key_id == "env_key_id"

    @patch.dict("os.environ", {"KAFKA_BOOTSTRAP_SERVERS": "kafka.example.com:9092"})
    @pytest.mark.asyncio
    async def test_kafka_configuration_from_env(self):
        """Test Kafka configuration from environment."""
        with patch(
            "omninode_bridge.services.metadata_stamping.events.publisher.KafkaClient"
        ) as mock_client_class:
            mock_kafka_client = AsyncMock()
            mock_client_class.return_value = mock_kafka_client
            mock_kafka_client.connect.return_value = None
            mock_kafka_client.is_connected = True

            publisher = EventPublisher(enable_events=True)
            await publisher.initialize()

            # Verify KafkaClient was created with correct config
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args[1]
            assert call_kwargs["bootstrap_servers"] == "kafka.example.com:9092"
            assert call_kwargs["enable_dead_letter_queue"] is True
            assert call_kwargs["max_retry_attempts"] == 3
