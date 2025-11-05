"""
Comprehensive tests for Kafka Topic Registry.

Tests topic registration, validation, naming conventions,
and environment prefixes.
"""

import pytest
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from omnibase_infra.infrastructure.kafka.topic_registry import KafkaTopicRegistry
from omnibase_infra.models.kafka import ModelTopicMetadata


class TestKafkaTopicRegistryInit:
    """Test topic registry initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        registry = KafkaTopicRegistry()

        assert registry.environment == "dev"
        assert registry.prefix_separator == "."
        assert len(registry.topics) == 0
        assert registry.max_topic_length == 249

    def test_init_custom_environment(self):
        """Test initialization with custom environment."""
        registry = KafkaTopicRegistry(environment="prod")

        assert registry.environment == "prod"

    def test_init_custom_separator(self):
        """Test initialization with custom separator."""
        registry = KafkaTopicRegistry(prefix_separator="-")

        assert registry.prefix_separator == "-"

    def test_valid_environments(self):
        """Test valid environment list."""
        registry = KafkaTopicRegistry()

        assert "dev" in registry.valid_environments
        assert "staging" in registry.valid_environments
        assert "prod" in registry.valid_environments
        assert "test" in registry.valid_environments
        assert "local" in registry.valid_environments


class TestKafkaTopicRegistryBuildTopicName:
    """Test topic name building."""

    def test_build_topic_name(self):
        """Test building fully qualified topic name."""
        registry = KafkaTopicRegistry(environment="dev")

        topic_name = registry.build_topic_name("user.events")

        assert topic_name == "dev.user.events"

    def test_build_topic_name_production(self):
        """Test building topic name in production."""
        registry = KafkaTopicRegistry(environment="prod")

        topic_name = registry.build_topic_name("orders.created")

        assert topic_name == "prod.orders.created"

    def test_build_topic_name_custom_separator(self):
        """Test building topic name with custom separator."""
        registry = KafkaTopicRegistry(environment="staging", prefix_separator="-")

        topic_name = registry.build_topic_name("audit.logs")

        assert topic_name == "staging-audit.logs"


class TestKafkaTopicRegistryParseTopicName:
    """Test topic name parsing."""

    def test_parse_topic_name_with_prefix(self):
        """Test parsing topic name with valid environment prefix."""
        registry = KafkaTopicRegistry(environment="dev")

        env, base_name = registry.parse_topic_name("prod.user.events")

        assert env == "prod"
        assert base_name == "user.events"

    def test_parse_topic_name_no_prefix(self):
        """Test parsing topic name without environment prefix."""
        registry = KafkaTopicRegistry(environment="dev")

        env, base_name = registry.parse_topic_name("user.events")

        assert env == "dev"  # Falls back to registry environment
        assert base_name == "user.events"

    def test_parse_topic_name_invalid_prefix(self):
        """Test parsing topic name with invalid environment prefix."""
        registry = KafkaTopicRegistry(environment="dev")

        env, base_name = registry.parse_topic_name("invalid.user.events")

        assert env == "dev"  # Falls back to registry environment
        assert base_name == "invalid.user.events"

    def test_parse_topic_name_custom_separator(self):
        """Test parsing with custom separator."""
        registry = KafkaTopicRegistry(environment="dev", prefix_separator="-")

        env, base_name = registry.parse_topic_name("prod-orders.created")

        assert env == "prod"
        assert base_name == "orders.created"


class TestKafkaTopicRegistryRegisterTopic:
    """Test topic registration."""

    def test_register_topic_basic(self):
        """Test basic topic registration."""
        registry = KafkaTopicRegistry(environment="dev")

        metadata = registry.register_topic(
            base_name="user.events",
            partition_count=3,
            replication_factor=2,
        )

        assert isinstance(metadata, ModelTopicMetadata)
        assert metadata.topic_name == "dev.user.events"
        assert metadata.base_name == "user.events"
        assert metadata.environment == "dev"
        assert metadata.partition_count == 3
        assert metadata.replication_factor == 2
        assert "dev.user.events" in registry.topics

    def test_register_topic_with_metadata(self):
        """Test topic registration with full metadata."""
        registry = KafkaTopicRegistry(environment="prod")

        metadata = registry.register_topic(
            base_name="orders.created",
            partition_count=6,
            replication_factor=3,
            description="Order creation events",
            owner="orders-team",
            tags=["orders", "critical"],
        )

        assert metadata.description == "Order creation events"
        assert metadata.owner == "orders-team"
        assert "orders" in metadata.tags
        assert "critical" in metadata.tags

    def test_register_topic_validation(self):
        """Test topic name validation during registration."""
        registry = KafkaTopicRegistry(environment="dev")

        metadata = registry.register_topic(
            base_name="valid.topic.name",
            partition_count=1,
        )

        # Validation should pass for valid names
        assert metadata.is_valid is True
        assert len(metadata.validation_errors) == 0

    def test_register_multiple_topics(self):
        """Test registering multiple topics."""
        registry = KafkaTopicRegistry(environment="staging")

        registry.register_topic("topic1")
        registry.register_topic("topic2")
        registry.register_topic("topic3")

        assert len(registry.topics) == 3
        assert "staging.topic1" in registry.topics
        assert "staging.topic2" in registry.topics
        assert "staging.topic3" in registry.topics

    def test_register_topic_updates_existing(self):
        """Test registering same topic updates metadata."""
        registry = KafkaTopicRegistry(environment="dev")

        # Register first time
        metadata1 = registry.register_topic(
            base_name="test.topic",
            partition_count=1,
            description="First registration",
        )

        # Register again with different metadata
        metadata2 = registry.register_topic(
            base_name="test.topic",
            partition_count=3,
            description="Updated registration",
        )

        # Should update the existing entry
        assert len(registry.topics) == 1
        assert metadata2.partition_count == 3
        assert metadata2.description == "Updated registration"


class TestKafkaTopicRegistryGetTopic:
    """Test topic retrieval."""

    def test_get_topic_exists(self):
        """Test retrieving existing topic."""
        registry = KafkaTopicRegistry(environment="dev")
        registry.register_topic("user.events")

        metadata = registry.get_topic("dev.user.events")

        assert metadata.topic_name == "dev.user.events"
        assert metadata.base_name == "user.events"

    def test_get_topic_not_found(self):
        """Test retrieving non-existent topic."""
        registry = KafkaTopicRegistry(environment="dev")

        with pytest.raises(OnexError) as exc_info:
            registry.get_topic("dev.missing.topic")

        assert exc_info.value.code == CoreErrorCode.NOT_FOUND


class TestKafkaTopicRegistryListTopics:
    """Test topic listing."""

    def test_list_all_topics(self):
        """Test listing all topics."""
        registry = KafkaTopicRegistry(environment="dev")
        registry.register_topic("topic1")
        registry.register_topic("topic2")
        registry.register_topic("topic3")

        topics = registry.list_topics()

        assert len(topics) == 3
        topic_names = [t.topic_name for t in topics]
        assert "dev.topic1" in topic_names
        assert "dev.topic2" in topic_names
        assert "dev.topic3" in topic_names

    def test_list_topics_by_environment(self):
        """Test filtering topics by environment."""
        registry = KafkaTopicRegistry(environment="dev")

        # Register topics in dev
        registry.register_topic("topic1")
        registry.register_topic("topic2")

        # Create registry for prod and register topics
        prod_registry = KafkaTopicRegistry(environment="prod")
        prod_registry.register_topic("topic3")

        dev_topics = registry.list_topics(environment="dev")

        assert len(dev_topics) == 2
        assert all(t.environment == "dev" for t in dev_topics)

    def test_list_topics_by_tag(self):
        """Test filtering topics by tag."""
        registry = KafkaTopicRegistry(environment="dev")
        registry.register_topic("topic1", tags=["critical", "user"])
        registry.register_topic("topic2", tags=["critical", "order"])
        registry.register_topic("topic3", tags=["audit"])

        critical_topics = registry.list_topics(tags=["critical"])

        assert len(critical_topics) == 2

    def test_list_topics_empty_registry(self):
        """Test listing topics from empty registry."""
        registry = KafkaTopicRegistry(environment="dev")

        topics = registry.list_topics()

        assert len(topics) == 0


class TestKafkaTopicRegistryValidation:
    """Test topic name validation."""

    def test_validate_valid_topic_name(self):
        """Test validation of valid topic name."""
        registry = KafkaTopicRegistry(environment="dev")

        metadata = registry.register_topic("valid.topic.name")

        assert metadata.is_valid is True
        assert len(metadata.validation_errors) == 0

    def test_validate_topic_name_too_long(self):
        """Test validation of excessively long topic name."""
        registry = KafkaTopicRegistry(environment="dev")

        # Create a very long topic name
        long_name = "a" * 250

        metadata = registry.register_topic(long_name)

        # Should have validation error
        assert metadata.is_valid is False
        assert len(metadata.validation_errors) > 0

    def test_validate_topic_name_with_special_chars(self):
        """Test validation of topic names with special characters."""
        registry = KafkaTopicRegistry(environment="dev")

        # Kafka allows alphanumeric, dots, hyphens, underscores
        valid_names = [
            "topic-name",
            "topic_name",
            "topic.name",
            "topic123",
        ]

        for name in valid_names:
            metadata = registry.register_topic(name)
            assert metadata.is_valid is True

    def test_validate_empty_topic_name(self):
        """Test validation of empty topic name."""
        registry = KafkaTopicRegistry(environment="dev")

        with pytest.raises(OnexError):
            registry.register_topic("")


class TestKafkaTopicRegistryTopicExists:
    """Test topic existence checks."""

    def test_topic_exists_true(self):
        """Test checking if topic exists."""
        registry = KafkaTopicRegistry(environment="dev")
        registry.register_topic("test.topic")

        exists = registry.topic_exists("dev.test.topic")

        assert exists is True

    def test_topic_exists_false(self):
        """Test checking non-existent topic."""
        registry = KafkaTopicRegistry(environment="dev")

        exists = registry.topic_exists("dev.missing.topic")

        assert exists is False

    def test_topic_exists_by_base_name(self):
        """Test checking topic existence by base name."""
        registry = KafkaTopicRegistry(environment="dev")
        registry.register_topic("test.topic")

        # Check using base name
        full_name = registry.build_topic_name("test.topic")
        exists = registry.topic_exists(full_name)

        assert exists is True


class TestKafkaTopicRegistryDeleteTopic:
    """Test topic deletion."""

    def test_delete_topic_success(self):
        """Test successful topic deletion."""
        registry = KafkaTopicRegistry(environment="dev")
        registry.register_topic("test.topic")

        assert registry.topic_exists("dev.test.topic")

        registry.delete_topic("dev.test.topic")

        assert not registry.topic_exists("dev.test.topic")

    def test_delete_topic_not_found(self):
        """Test deleting non-existent topic."""
        registry = KafkaTopicRegistry(environment="dev")

        with pytest.raises(OnexError) as exc_info:
            registry.delete_topic("dev.missing.topic")

        assert exc_info.value.code == CoreErrorCode.NOT_FOUND


class TestKafkaTopicRegistryIntegration:
    """Integration tests for complete workflows."""

    def test_full_topic_lifecycle(self):
        """Test complete topic lifecycle."""
        registry = KafkaTopicRegistry(environment="prod")

        # Register topic
        metadata = registry.register_topic(
            base_name="orders.created",
            partition_count=6,
            replication_factor=3,
            description="Order events",
            owner="orders-team",
            tags=["orders", "critical"],
        )

        assert metadata.is_valid is True

        # Retrieve topic
        retrieved = registry.get_topic("prod.orders.created")
        assert retrieved.base_name == "orders.created"
        assert retrieved.partition_count == 6

        # List topics
        topics = registry.list_topics()
        assert len(topics) == 1

        # Delete topic
        registry.delete_topic("prod.orders.created")
        assert not registry.topic_exists("prod.orders.created")

    def test_multiple_environments(self):
        """Test working with multiple environments."""
        dev_registry = KafkaTopicRegistry(environment="dev")
        prod_registry = KafkaTopicRegistry(environment="prod")

        # Register in different environments
        dev_registry.register_topic("test.topic")
        prod_registry.register_topic("test.topic")

        dev_topics = dev_registry.list_topics()
        prod_topics = prod_registry.list_topics()

        assert len(dev_topics) == 1
        assert len(prod_topics) == 1
        assert dev_topics[0].topic_name == "dev.test.topic"
        assert prod_topics[0].topic_name == "prod.test.topic"

    def test_topic_name_parsing_consistency(self):
        """Test consistency between build and parse operations."""
        registry = KafkaTopicRegistry(environment="staging")

        base_name = "user.events"
        built_name = registry.build_topic_name(base_name)
        parsed_env, parsed_base = registry.parse_topic_name(built_name)

        assert parsed_env == "staging"
        assert parsed_base == base_name
