"""Tests for event schema versioning."""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from omninode_bridge.events.versioning import (
    EventSchemaVersion,
    EventVersionRegistry,
    SchemaEvolutionStrategy,
    get_topic_name,
    parse_topic_name,
)


# Test event schemas
class TestEventV1(BaseModel):
    """Test event schema V1."""

    event_type: Literal["TEST_EVENT"] = "TEST_EVENT"
    field_a: str
    field_b: int


class TestEventV2(BaseModel):
    """Test event schema V2 with optional new field."""

    event_type: Literal["TEST_EVENT"] = "TEST_EVENT"
    field_a: str
    field_b: int
    field_c: str = Field(default="default")  # New optional field


class TestEventVersionRegistry:
    """Tests for EventVersionRegistry."""

    def test_register_schema(self):
        """Test registering event schema."""
        registry = EventVersionRegistry()
        registry.register(
            "TEST_EVENT", EventSchemaVersion.V1, TestEventV1, deprecated=False
        )

        schema = registry.get_schema("TEST_EVENT", EventSchemaVersion.V1)
        assert schema == TestEventV1

    def test_get_schema_raises_on_missing_event(self):
        """Test get_schema raises KeyError for missing event."""
        registry = EventVersionRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_schema("MISSING_EVENT", EventSchemaVersion.V1)

    def test_get_schema_raises_on_missing_version(self):
        """Test get_schema raises KeyError for missing version."""
        registry = EventVersionRegistry()
        registry.register("TEST_EVENT", EventSchemaVersion.V1, TestEventV1)

        with pytest.raises(KeyError, match="not found"):
            registry.get_schema("TEST_EVENT", EventSchemaVersion.V2)

    def test_get_metadata(self):
        """Test getting schema metadata."""
        registry = EventVersionRegistry()
        registry.register(
            "TEST_EVENT",
            EventSchemaVersion.V1,
            TestEventV1,
            evolution_strategy=SchemaEvolutionStrategy.BACKWARD_COMPATIBLE,
            deprecated=False,
        )

        metadata = registry.get_metadata("TEST_EVENT", EventSchemaVersion.V1)
        assert metadata.version == EventSchemaVersion.V1
        assert metadata.schema_class == TestEventV1
        assert (
            metadata.evolution_strategy == SchemaEvolutionStrategy.BACKWARD_COMPATIBLE
        )
        assert metadata.deprecated is False

    def test_get_latest_version(self):
        """Test getting latest version."""
        registry = EventVersionRegistry()
        registry.register("TEST_EVENT", EventSchemaVersion.V1, TestEventV1)
        registry.register("TEST_EVENT", EventSchemaVersion.V2, TestEventV2)

        latest = registry.get_latest_version("TEST_EVENT")
        assert latest == EventSchemaVersion.V2

    def test_get_latest_version_ignores_deprecated(self):
        """Test that latest version ignores deprecated versions."""
        registry = EventVersionRegistry()
        registry.register("TEST_EVENT", EventSchemaVersion.V1, TestEventV1)
        registry.register(
            "TEST_EVENT", EventSchemaVersion.V2, TestEventV2, deprecated=True
        )

        latest = registry.get_latest_version("TEST_EVENT")
        assert latest == EventSchemaVersion.V1

    def test_register_migration(self):
        """Test registering migration function."""
        registry = EventVersionRegistry()

        def migrate_v1_to_v2(data: dict) -> dict:
            data["field_c"] = "migrated"
            return data

        registry.register_migration(
            "TEST_EVENT", EventSchemaVersion.V1, EventSchemaVersion.V2, migrate_v1_to_v2
        )

        # Migration should be registered
        assert "TEST_EVENT" in registry._migrations

    def test_migrate(self):
        """Test migrating event data."""
        registry = EventVersionRegistry()

        def migrate_v1_to_v2(data: dict) -> dict:
            data["field_c"] = "migrated"
            return data

        registry.register_migration(
            "TEST_EVENT", EventSchemaVersion.V1, EventSchemaVersion.V2, migrate_v1_to_v2
        )

        data = {"event_type": "TEST_EVENT", "field_a": "test", "field_b": 123}
        migrated = registry.migrate(
            "TEST_EVENT", data, EventSchemaVersion.V1, EventSchemaVersion.V2
        )

        assert migrated["field_c"] == "migrated"

    def test_migrate_same_version_returns_data(self):
        """Test that migration with same version returns original data."""
        registry = EventVersionRegistry()
        data = {"event_type": "TEST_EVENT", "field_a": "test", "field_b": 123}

        result = registry.migrate(
            "TEST_EVENT", data, EventSchemaVersion.V1, EventSchemaVersion.V1
        )

        assert result == data

    def test_migrate_raises_on_missing_migration(self):
        """Test that migrate raises KeyError when no migration path."""
        registry = EventVersionRegistry()
        data = {"event_type": "TEST_EVENT", "field_a": "test", "field_b": 123}

        with pytest.raises(KeyError, match="No migration path"):
            registry.migrate(
                "TEST_EVENT", data, EventSchemaVersion.V1, EventSchemaVersion.V2
            )

    def test_validate_and_migrate(self):
        """Test validating and migrating event data."""
        registry = EventVersionRegistry()
        registry.register("TEST_EVENT", EventSchemaVersion.V1, TestEventV1)
        registry.register("TEST_EVENT", EventSchemaVersion.V2, TestEventV2)

        def migrate_v1_to_v2(data: dict) -> dict:
            data["field_c"] = "migrated"
            return data

        registry.register_migration(
            "TEST_EVENT", EventSchemaVersion.V1, EventSchemaVersion.V2, migrate_v1_to_v2
        )

        data = {"event_type": "TEST_EVENT", "field_a": "test", "field_b": 123}
        validated = registry.validate_and_migrate(
            "TEST_EVENT", data, EventSchemaVersion.V1, EventSchemaVersion.V2
        )

        assert isinstance(validated, TestEventV2)
        assert validated.field_c == "migrated"

    def test_validate_and_migrate_uses_latest_by_default(self):
        """Test that validate_and_migrate uses latest version by default."""
        registry = EventVersionRegistry()
        registry.register("TEST_EVENT", EventSchemaVersion.V1, TestEventV1)
        registry.register("TEST_EVENT", EventSchemaVersion.V2, TestEventV2)

        def migrate_v1_to_v2(data: dict) -> dict:
            data["field_c"] = "default_migration"
            return data

        registry.register_migration(
            "TEST_EVENT", EventSchemaVersion.V1, EventSchemaVersion.V2, migrate_v1_to_v2
        )

        data = {"event_type": "TEST_EVENT", "field_a": "test", "field_b": 123}
        validated = registry.validate_and_migrate(
            "TEST_EVENT", data, EventSchemaVersion.V1
        )

        # Should migrate to V2 (latest)
        assert isinstance(validated, TestEventV2)
        assert validated.field_c == "default_migration"

    def test_list_event_types(self):
        """Test listing registered event types."""
        registry = EventVersionRegistry()
        registry.register("EVENT_A", EventSchemaVersion.V1, TestEventV1)
        registry.register("EVENT_B", EventSchemaVersion.V1, TestEventV1)

        event_types = registry.list_event_types()
        assert "EVENT_A" in event_types
        assert "EVENT_B" in event_types

    def test_list_versions(self):
        """Test listing versions for event type."""
        registry = EventVersionRegistry()
        registry.register("TEST_EVENT", EventSchemaVersion.V1, TestEventV1)
        registry.register("TEST_EVENT", EventSchemaVersion.V2, TestEventV2)

        versions = registry.list_versions("TEST_EVENT")
        assert EventSchemaVersion.V1 in versions
        assert EventSchemaVersion.V2 in versions

    def test_is_deprecated(self):
        """Test checking if version is deprecated."""
        registry = EventVersionRegistry()
        registry.register(
            "TEST_EVENT", EventSchemaVersion.V1, TestEventV1, deprecated=True
        )
        registry.register("TEST_EVENT", EventSchemaVersion.V2, TestEventV2)

        assert registry.is_deprecated("TEST_EVENT", EventSchemaVersion.V1) is True
        assert registry.is_deprecated("TEST_EVENT", EventSchemaVersion.V2) is False


class TestTopicNaming:
    """Tests for topic naming functions."""

    def test_get_topic_name_default(self):
        """Test get_topic_name with default parameters."""
        topic = get_topic_name("generation-requested")
        assert topic == "dev.omninode-bridge.codegen.generation-requested.v1"

    def test_get_topic_name_custom_version(self):
        """Test get_topic_name with custom version."""
        topic = get_topic_name("generation-requested", version=EventSchemaVersion.V2)
        assert topic == "dev.omninode-bridge.codegen.generation-requested.v2"

    def test_get_topic_name_custom_environment(self):
        """Test get_topic_name with custom environment."""
        topic = get_topic_name("generation-requested", environment="prod")
        assert topic == "prod.omninode-bridge.codegen.generation-requested.v1"

    def test_get_topic_name_custom_service(self):
        """Test get_topic_name with custom service."""
        topic = get_topic_name("generation-requested", service="omniarchon")
        assert topic == "dev.omniarchon.codegen.generation-requested.v1"

    def test_get_topic_name_all_custom(self):
        """Test get_topic_name with all custom parameters."""
        topic = get_topic_name(
            "custom-event",
            version=EventSchemaVersion.V2,
            environment="staging",
            service="custom-service",
            domain="custom-domain",
        )
        assert topic == "staging.custom-service.custom-domain.custom-event.v2"

    def test_parse_topic_name(self):
        """Test parsing topic name."""
        topic = "dev.omninode-bridge.codegen.generation-requested.v1"
        parsed = parse_topic_name(topic)

        assert parsed["environment"] == "dev"
        assert parsed["service"] == "omninode-bridge"
        assert parsed["domain"] == "codegen"
        assert parsed["base_name"] == "generation-requested"
        assert parsed["version"] == "v1"

    def test_parse_topic_name_multi_part_base(self):
        """Test parsing topic name with multi-part base name."""
        topic = "prod.service.domain.multi.part.base.v2"
        parsed = parse_topic_name(topic)

        assert parsed["environment"] == "prod"
        assert parsed["service"] == "service"
        assert parsed["domain"] == "domain"
        assert parsed["base_name"] == "multi.part.base"
        assert parsed["version"] == "v2"

    def test_parse_topic_name_invalid_format(self):
        """Test parse_topic_name raises on invalid format."""
        with pytest.raises(ValueError, match="Invalid topic name format"):
            parse_topic_name("invalid.topic")


class TestSchemaEvolution:
    """Tests for schema evolution scenarios."""

    def test_backward_compatible_evolution(self):
        """Test backward compatible schema evolution."""
        registry = EventVersionRegistry()
        registry.register(
            "TEST_EVENT",
            EventSchemaVersion.V1,
            TestEventV1,
            evolution_strategy=SchemaEvolutionStrategy.BACKWARD_COMPATIBLE,
        )
        registry.register(
            "TEST_EVENT",
            EventSchemaVersion.V2,
            TestEventV2,
            evolution_strategy=SchemaEvolutionStrategy.BACKWARD_COMPATIBLE,
        )

        # V1 data should validate with V2 schema (new field is optional)
        v1_data = {"event_type": "TEST_EVENT", "field_a": "test", "field_b": 123}
        v2_instance = TestEventV2(**v1_data)

        assert v2_instance.field_a == "test"
        assert v2_instance.field_b == 123
        assert v2_instance.field_c == "default"  # Uses default value

    def test_deprecation_metadata(self):
        """Test schema deprecation metadata."""
        registry = EventVersionRegistry()
        registry.register(
            "TEST_EVENT",
            EventSchemaVersion.V1,
            TestEventV1,
            deprecated=True,
            deprecation_date="2025-01-01",
            removal_date="2025-06-01",
            migration_notes="Migrate to V2 using migrate_v1_to_v2",
        )

        metadata = registry.get_metadata("TEST_EVENT", EventSchemaVersion.V1)
        assert metadata.deprecated is True
        assert metadata.deprecation_date == "2025-01-01"
        assert metadata.removal_date == "2025-06-01"
        assert "V2" in metadata.migration_notes
