#!/usr/bin/env python3
"""Unit tests for KafkaConsumerWrapper.

Tests cover:
- Initialization with and without aiokafka
- Configuration from environment and parameters
- Topic name building
- Subscription to topics
- Message consumption
- Offset committing
- Consumer closing
- Error handling
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError

# Import the module to test
from omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper import (
    AIOKAFKA_AVAILABLE,
    KafkaConsumerWrapper,
)


class TestKafkaConsumerWrapper:
    """Test suite for KafkaConsumerWrapper."""

    def test_init_with_aiokafka_unavailable(self):
        """Test initialization when aiokafka is not available."""
        with patch(
            "omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper.AIOKAFKA_AVAILABLE",
            False,
        ):
            with pytest.raises(ModelOnexError) as exc_info:
                KafkaConsumerWrapper()

            assert exc_info.value.error_code == EnumCoreErrorCode.DEPENDENCY_ERROR
            assert "aiokafka not installed" in str(exc_info.value)

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        bootstrap_servers = "kafka1:9092,kafka2:9092"
        security_protocol = "SSL"
        sasl_mechanism = "SCRAM-SHA-256"
        sasl_username = "test_user"
        sasl_password = "test_pass"

        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper(
                bootstrap_servers=bootstrap_servers,
                security_protocol=security_protocol,
                sasl_mechanism=sasl_mechanism,
                sasl_username=sasl_username,
                sasl_password=sasl_password,
            )

            assert consumer._bootstrap_servers == bootstrap_servers
            assert consumer._security_protocol == security_protocol
            assert consumer._sasl_mechanism == sasl_mechanism
            assert consumer._sasl_username == sasl_username
            assert consumer._sasl_password == sasl_password

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_init_with_environment_variables(self):
        """Test initialization with environment variables."""
        env_vars = {
            "KAFKA_BOOTSTRAP_SERVERS": "env-kafka:9092",
            "KAFKA_SECURITY_PROTOCOL": "SASL_SSL",
            "KAFKA_SASL_MECHANISM": "PLAIN",
            "KAFKA_SASL_USERNAME": "env_user",
            "KAFKA_SASL_PASSWORD": "env_pass",
            "OMNINODE_ENV": "production",
            "OMNINODE_TENANT": "omnibase",
            "OMNINODE_CONTEXT": "bridge",
        }

        with patch.dict(os.environ, env_vars):
            consumer = KafkaConsumerWrapper(security_protocol=None)

            assert consumer._bootstrap_servers == "env-kafka:9092"
            assert consumer._security_protocol == "SASL_SSL"
            assert consumer._sasl_mechanism == "PLAIN"
            assert consumer._sasl_username == "env_user"
            assert consumer._sasl_password == "env_pass"
            assert consumer._env == "production"
            assert consumer._tenant == "omnibase"
            assert consumer._context == "bridge"

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper()

            assert consumer._bootstrap_servers == "omninode-bridge-redpanda:9092"
            assert consumer._security_protocol == "PLAINTEXT"
            assert consumer._sasl_mechanism is None
            assert consumer._sasl_username is None
            assert consumer._sasl_password is None
            assert consumer._env == "dev"
            assert consumer._tenant == "omninode_bridge"
            assert consumer._context == "onex"

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_build_topic_name_default(self):
        """Test topic name building with default class."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper()
            topic = consumer._build_topic_name("workflow-started")

            assert topic == "dev.omninode_bridge.onex.evt.workflow-started.v1"

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_build_topic_name_custom_class(self):
        """Test topic name building with custom class."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper()
            topic = consumer._build_topic_name("create-user", "cmd")

            assert topic == "dev.omninode_bridge.onex.cmd.create-user.v1"

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_build_security_config_plaintext(self):
        """Test security config building for PLAINTEXT."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper(security_protocol="PLAINTEXT")
            config = consumer._build_security_config()

            assert config == {}

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_build_security_config_ssl(self):
        """Test security config building for SSL."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper(security_protocol="SSL")
            config = consumer._build_security_config()

            assert config["security_protocol"] == "SSL"

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_build_security_config_sasl(self):
        """Test security config building for SASL."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper(
                security_protocol="SASL_SSL",
                sasl_mechanism="SCRAM-SHA-256",
                sasl_username="test_user",
                sasl_password="test_pass",
            )
            config = consumer._build_security_config()

            assert config["security_protocol"] == "SASL_SSL"
            assert config["sasl_mechanism"] == "SCRAM-SHA-256"
            assert config["sasl_plain_username"] == "test_user"
            assert config["sasl_plain_password"] == "test_pass"

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_subscribe_to_topics_success(self):
        """Test successful topic subscription."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper.AIOKafkaConsumer"
            ) as mock_consumer_class:
                # Setup mock consumer
                mock_consumer = AsyncMock()
                mock_consumer_class.return_value = mock_consumer

                consumer = KafkaConsumerWrapper()
                topics = ["workflow-started", "metadata-stamp-created"]
                group_id = "test_group"

                await consumer.subscribe_to_topics(topics, group_id)

                # Verify consumer was created with correct parameters
                mock_consumer_class.assert_called_once()
                call_args = mock_consumer_class.call_args[0]
                assert list(call_args) == [
                    "dev.omninode_bridge.onex.evt.workflow-started.v1",
                    "dev.omninode_bridge.onex.evt.metadata-stamp-created.v1",
                ]
                call_kwargs = mock_consumer_class.call_args[1]
                assert call_kwargs["bootstrap_servers"] == [
                    "omninode-bridge-redpanda:9092"
                ]
                assert call_kwargs["group_id"] == group_id
                assert call_kwargs["auto_offset_reset"] == "latest"
                assert call_kwargs["enable_auto_commit"] is False

                # Verify consumer methods were called
                mock_consumer.start.assert_called_once()

                # Verify state was updated
                assert consumer._consumer == mock_consumer
                assert consumer._subscribed_topics == [
                    "dev.omninode_bridge.onex.evt.workflow-started.v1",
                    "dev.omninode_bridge.onex.evt.metadata-stamp-created.v1",
                ]
                assert consumer._consumer_group == group_id
                assert consumer._is_running is True

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_subscribe_to_topics_already_subscribed(self):
        """Test subscription when consumer already subscribed."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper.AIOKafkaConsumer"
            ) as mock_consumer_class:
                # Setup mock consumer
                mock_consumer = AsyncMock()
                mock_consumer_class.return_value = mock_consumer

                consumer = KafkaConsumerWrapper()

                # Subscribe once
                await consumer.subscribe_to_topics(["test-topic"], "test_group")

                # Try to subscribe again
                with pytest.raises(ModelOnexError) as exc_info:
                    await consumer.subscribe_to_topics(
                        ["another-topic"], "another_group"
                    )

                assert exc_info.value.error_code == EnumCoreErrorCode.VALIDATION_FAILED
                assert "already subscribed" in str(exc_info.value)

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_subscribe_to_topics_kafka_error(self):
        """Test subscription when Kafka raises an error."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper.AIOKafkaConsumer"
            ) as mock_consumer_class:
                # Setup mock consumer to raise error
                mock_consumer_class.side_effect = Exception("Kafka connection failed")

                consumer = KafkaConsumerWrapper()

                with pytest.raises(ModelOnexError) as exc_info:
                    await consumer.subscribe_to_topics(["test-topic"], "test_group")

                assert exc_info.value.error_code == EnumCoreErrorCode.INTERNAL_ERROR
                assert "Unexpected error subscribing" in str(exc_info.value)

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_consume_messages_stream_success(self):
        """Test successful message consumption with batch processing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper.AIOKafkaConsumer"
            ) as mock_consumer_class:
                # Setup mock consumer
                mock_consumer = AsyncMock()
                mock_consumer_class.return_value = mock_consumer

                # Setup mock message
                mock_message = MagicMock()
                mock_message.key = "test-key"
                mock_message.value = {"test": "data"}
                mock_message.topic = "test-topic"
                mock_message.partition = 0
                mock_message.offset = 123
                mock_message.timestamp = 1234567890
                mock_message.headers = {"header1": "value1"}

                # Setup mock TopicPartition for getmany() response
                from aiokafka import TopicPartition

                mock_topic_partition = TopicPartition("test-topic", 0)

                # Mock getmany() to return batch of messages
                # First call returns messages, second call returns empty (to allow break)
                mock_consumer.getmany.side_effect = [
                    {mock_topic_partition: [mock_message]},  # First batch
                    {},  # Second batch (empty, triggers continue loop)
                ]

                consumer = KafkaConsumerWrapper()
                await consumer.subscribe_to_topics(["test-topic"], "test_group")

                # Consume messages
                messages = []
                async for batch in consumer.consume_messages_stream():
                    messages.extend(batch)
                    break  # Only consume first batch

                # Verify message was processed correctly
                assert len(messages) == 1
                message = messages[0]
                assert message["key"] == "test-key"
                assert message["value"] == {"test": "data"}
                assert message["topic"] == "test-topic"
                assert message["partition"] == 0
                assert message["offset"] == 123
                assert message["timestamp"] == 1234567890
                assert message["headers"] == {"header1": "value1"}

                # Verify getmany was called with correct parameters
                assert mock_consumer.getmany.call_count >= 1
                first_call = mock_consumer.getmany.call_args_list[0]
                assert first_call[1]["timeout_ms"] == 1000  # default
                assert first_call[1]["max_records"] == 500  # default

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_consume_messages_stream_not_subscribed(self):
        """Test message consumption when not subscribed."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper()

            with pytest.raises(ModelOnexError) as exc_info:
                async for _ in consumer.consume_messages_stream():
                    pass

            assert exc_info.value.error_code == EnumCoreErrorCode.VALIDATION_FAILED
            assert "not subscribed" in str(exc_info.value)

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_commit_offsets_success(self):
        """Test successful offset commit."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper.AIOKafkaConsumer"
            ) as mock_consumer_class:
                # Setup mock consumer
                mock_consumer = AsyncMock()
                mock_consumer_class.return_value = mock_consumer

                consumer = KafkaConsumerWrapper()
                await consumer.subscribe_to_topics(["test-topic"], "test_group")

                # Commit offsets
                await consumer.commit_offsets()

                # Verify commit was called
                mock_consumer.commit.assert_called_once()

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_commit_offsets_not_subscribed(self):
        """Test offset commit when not subscribed."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper()

            with pytest.raises(ModelOnexError) as exc_info:
                await consumer.commit_offsets()

            assert exc_info.value.error_code == EnumCoreErrorCode.VALIDATION_FAILED
            assert "not subscribed" in str(exc_info.value)

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_close_consumer_success(self):
        """Test successful consumer closing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper.AIOKafkaConsumer"
            ) as mock_consumer_class:
                # Setup mock consumer
                mock_consumer = AsyncMock()
                mock_consumer_class.return_value = mock_consumer

                consumer = KafkaConsumerWrapper()
                await consumer.subscribe_to_topics(["test-topic"], "test_group")

                # Close consumer
                await consumer.close_consumer()

                # Verify cleanup methods were called
                mock_consumer.commit.assert_called_once()
                mock_consumer.stop.assert_called_once()

                # Verify state was reset
                assert consumer._consumer is None
                assert consumer._subscribed_topics == []
                assert consumer._consumer_group is None
                assert consumer._is_running is False

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    @pytest.mark.asyncio
    async def test_close_consumer_not_started(self):
        """Test consumer closing when never started."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper()

            # Close consumer (should not raise)
            await consumer.close_consumer()

            # Verify state remains reset
            assert consumer._consumer is None
            assert consumer._subscribed_topics == []
            assert consumer._consumer_group is None
            assert consumer._is_running is False

    @pytest.mark.skipif(not AIOKAFKA_AVAILABLE, reason="aiokafka not available")
    def test_properties(self):
        """Test consumer properties."""
        with patch.dict(os.environ, {}, clear=True):
            consumer = KafkaConsumerWrapper()

            # Test initial state
            assert consumer.is_subscribed is False
            assert consumer.subscribed_topics == []
            assert consumer.consumer_group is None
