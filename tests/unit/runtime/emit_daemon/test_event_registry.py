# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EventRegistry - Event type to Kafka topic mapping.

Tests the generic event registry that maps semantic event types to Kafka topics
and handles metadata injection.

Test coverage includes:
- Empty-by-default behavior
- Event type registration (single and batch)
- Topic resolution
- Partition key extraction
- Payload validation
- Metadata injection with deterministic mocking
- Realm-agnostic topic resolution
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_core.errors import OnexError
from omnibase_infra.runtime.emit_daemon.event_registry import (
    EventRegistry,
    ModelEventRegistration,
)
from omnibase_infra.runtime.emit_daemon.topics import PHASE_METRICS_REGISTRATION

# Fixed values for deterministic tests
FIXED_UUID = UUID("12345678-1234-5678-1234-567812345678")
FIXED_TIMESTAMP = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
FIXED_ISO_TIMESTAMP = "2025-01-15T10:30:00+00:00"

# Reusable sample registration for tests that need a populated registry
SAMPLE_REGISTRATION = ModelEventRegistration(
    event_type="test.event",
    topic_template="onex.evt.test.topic.v1",
    partition_key_field="session_id",
    required_fields=("field_a",),
    schema_version="1.0.0",
)


@pytest.fixture
def populated_registry() -> EventRegistry:
    """Registry pre-loaded with a sample event type for testing generic behavior."""
    registry = EventRegistry(environment="dev")
    registry.register(SAMPLE_REGISTRATION)
    return registry


@pytest.mark.unit
class TestModelEventRegistration:
    """Tests for ModelEventRegistration Pydantic model."""

    def test_minimal_registration(self) -> None:
        """Should create registration with only required fields."""
        reg = ModelEventRegistration(
            event_type="test.event",
            topic_template="onex.evt.test.topic.v1",
        )
        assert reg.event_type == "test.event"
        assert reg.topic_template == "onex.evt.test.topic.v1"
        assert reg.partition_key_field is None
        assert reg.required_fields == ()
        assert reg.schema_version == "1.0.0"

    def test_full_registration(self) -> None:
        """Should create registration with all fields specified."""
        reg = ModelEventRegistration(
            event_type="custom.event",
            topic_template="onex.evt.custom.topic.v2",
            partition_key_field="user_id",
            required_fields=("user_id", "action"),
            schema_version="2.0.0",
        )
        assert reg.event_type == "custom.event"
        assert reg.topic_template == "onex.evt.custom.topic.v2"
        assert reg.partition_key_field == "user_id"
        assert reg.required_fields == ("user_id", "action")
        assert reg.schema_version == "2.0.0"

    def test_registration_is_frozen(self) -> None:
        """Should raise when attempting to modify frozen model."""
        reg = ModelEventRegistration(
            event_type="test.event",
            topic_template="onex.evt.test.topic.v1",
        )
        with pytest.raises(Exception):  # ValidationError for frozen model
            reg.event_type = "modified.event"  # type: ignore[misc]

    def test_registration_forbids_extra_fields(self) -> None:
        """Should raise when extra fields are provided."""
        with pytest.raises(Exception):  # ValidationError for extra fields
            ModelEventRegistration(
                event_type="test.event",
                topic_template="onex.evt.test.topic.v1",
                unknown_field="value",  # type: ignore[call-arg]
            )


@pytest.mark.unit
class TestEventRegistryEmptyByDefault:
    """Tests that the registry starts with no registrations."""

    def test_fresh_registry_has_no_registrations(self) -> None:
        """A new EventRegistry should have zero event types registered."""
        registry = EventRegistry()
        assert registry.list_event_types() == []

    def test_fresh_registry_raises_on_resolve(self) -> None:
        """Resolving any event type on a fresh registry should raise OnexError."""
        registry = EventRegistry()
        with pytest.raises(OnexError, match="Unknown event type"):
            registry.resolve_topic("anything")

    def test_fresh_registry_get_registration_returns_none(self) -> None:
        """get_registration on a fresh registry should return None."""
        registry = EventRegistry()
        assert registry.get_registration("anything") is None


@pytest.mark.unit
class TestEventRegistryResolveTopic:
    """Tests for resolve_topic() method."""

    def test_resolve_topic_returns_registered_template(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should return the topic template for a registered event type."""
        topic = populated_registry.resolve_topic("test.event")
        assert topic == "onex.evt.test.topic.v1"

    def test_resolve_topic_returns_template_unchanged(self) -> None:
        """Should return topic template unchanged (realm-agnostic)."""
        registry = EventRegistry(environment="production")
        registry.register(
            ModelEventRegistration(
                event_type="custom.event",
                topic_template="onex.evt.custom.namespace.topic.v1",
            )
        )
        topic = registry.resolve_topic("custom.event")
        assert topic == "onex.evt.custom.namespace.topic.v1"

    def test_resolve_topic_raises_for_unknown_event_type(self) -> None:
        """Should raise OnexError for unknown event type."""
        registry = EventRegistry()
        with pytest.raises(OnexError, match=r"Unknown event type: 'unknown\.event'"):
            registry.resolve_topic("unknown.event")

    def test_resolve_topic_error_includes_registered_types(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should include registered types in error message."""
        with pytest.raises(OnexError) as exc_info:
            populated_registry.resolve_topic("unknown.event")
        error_message = str(exc_info.value)
        assert "test.event" in error_message


@pytest.mark.unit
class TestEventRegistryCustomRegistration:
    """Tests for register() method and custom registrations."""

    def test_register_adds_new_event_type(self) -> None:
        """Should add new event type to registry."""
        registry = EventRegistry()
        registration = ModelEventRegistration(
            event_type="custom.event",
            topic_template="onex.evt.custom.topic.v1",
        )
        registry.register(registration)

        topic = registry.resolve_topic("custom.event")
        assert topic == "onex.evt.custom.topic.v1"
        assert "custom.event" in registry.list_event_types()

    def test_register_can_override_existing(self) -> None:
        """Should allow overriding existing registrations."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="my.event",
                topic_template="onex.evt.original.v1",
            )
        )

        # Override with new topic
        registry.register(
            ModelEventRegistration(
                event_type="my.event",
                topic_template="onex.evt.overridden.v2",
            )
        )

        topic = registry.resolve_topic("my.event")
        assert topic == "onex.evt.overridden.v2"

    def test_register_multiple_custom_types(self) -> None:
        """Should allow registering multiple custom event types."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="custom.one",
                topic_template="onex.evt.custom.one.v1",
            )
        )
        registry.register(
            ModelEventRegistration(
                event_type="custom.two",
                topic_template="onex.evt.custom.two.v1",
            )
        )

        assert "custom.one" in registry.list_event_types()
        assert "custom.two" in registry.list_event_types()
        assert registry.resolve_topic("custom.one") == "onex.evt.custom.one.v1"
        assert registry.resolve_topic("custom.two") == "onex.evt.custom.two.v1"


@pytest.mark.unit
class TestEventRegistryRegisterBatch:
    """Tests for register_batch() method."""

    def test_register_batch_registers_all(self) -> None:
        """Should register all event types from the batch."""
        registry = EventRegistry()
        registrations = [
            ModelEventRegistration(
                event_type="batch.one",
                topic_template="onex.evt.batch.one.v1",
            ),
            ModelEventRegistration(
                event_type="batch.two",
                topic_template="onex.evt.batch.two.v1",
            ),
            ModelEventRegistration(
                event_type="batch.three",
                topic_template="onex.evt.batch.three.v1",
            ),
        ]
        registry.register_batch(registrations)

        assert len(registry.list_event_types()) == 3
        assert registry.resolve_topic("batch.one") == "onex.evt.batch.one.v1"
        assert registry.resolve_topic("batch.two") == "onex.evt.batch.two.v1"
        assert registry.resolve_topic("batch.three") == "onex.evt.batch.three.v1"

    def test_register_batch_empty_iterable(self) -> None:
        """Should be a no-op for empty iterable."""
        registry = EventRegistry()
        registry.register_batch([])
        assert registry.list_event_types() == []

    def test_register_batch_overrides_existing(self) -> None:
        """Should override existing registrations with same event type."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="my.event",
                topic_template="onex.evt.original.v1",
            )
        )

        registry.register_batch(
            [
                ModelEventRegistration(
                    event_type="my.event",
                    topic_template="onex.evt.overridden.v2",
                ),
                ModelEventRegistration(
                    event_type="new.event",
                    topic_template="onex.evt.new.v1",
                ),
            ]
        )

        assert registry.resolve_topic("my.event") == "onex.evt.overridden.v2"
        assert registry.resolve_topic("new.event") == "onex.evt.new.v1"

    def test_register_batch_accepts_generator(self) -> None:
        """Should accept any iterable, not just lists."""

        def gen() -> Iterable[ModelEventRegistration]:
            yield ModelEventRegistration(
                event_type="gen.one",
                topic_template="onex.evt.gen.one.v1",
            )
            yield ModelEventRegistration(
                event_type="gen.two",
                topic_template="onex.evt.gen.two.v1",
            )

        from collections.abc import Iterable

        registry = EventRegistry()
        registry.register_batch(gen())
        assert len(registry.list_event_types()) == 2


@pytest.mark.unit
class TestEventRegistryGetPartitionKey:
    """Tests for get_partition_key() method."""

    def test_extracts_partition_key_from_payload(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should extract partition key based on configured field."""
        key = populated_registry.get_partition_key(
            "test.event",
            {"field_a": "value", "session_id": "sess-abc123"},
        )
        assert key == "sess-abc123"

    def test_returns_none_when_field_not_in_payload(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should return None when partition key field is missing from payload."""
        key = populated_registry.get_partition_key(
            "test.event",
            {"field_a": "value"},  # No session_id
        )
        assert key is None

    def test_returns_none_when_no_partition_key_configured(self) -> None:
        """Should return None when no partition_key_field is configured."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="no.partition",
                topic_template="onex.evt.no.partition.v1",
                partition_key_field=None,
            )
        )
        key = registry.get_partition_key(
            "no.partition",
            {"data": "value", "session_id": "ignored"},
        )
        assert key is None

    def test_converts_non_string_partition_key_to_string(self) -> None:
        """Should convert non-string partition key values to string."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="numeric.id",
                topic_template="onex.evt.numeric.v1",
                partition_key_field="id",
            )
        )
        key = registry.get_partition_key("numeric.id", {"id": 12345})
        assert key == "12345"

    def test_partition_key_raises_for_unknown_event_type(self) -> None:
        """Should raise OnexError for unknown event type."""
        registry = EventRegistry()
        with pytest.raises(OnexError, match="Unknown event type"):
            registry.get_partition_key("unknown.event", {"data": "value"})

    def test_partition_key_returns_none_for_none_value(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should return None when partition key field value is None."""
        key = populated_registry.get_partition_key(
            "test.event",
            {"field_a": "value", "session_id": None},
        )
        assert key is None


@pytest.mark.unit
class TestEventRegistryValidatePayload:
    """Tests for validate_payload() method."""

    def test_returns_true_when_all_required_fields_present(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should return True when all required fields are present."""
        result = populated_registry.validate_payload(
            "test.event",
            {"field_a": "value"},
        )
        assert result is True

    def test_returns_true_with_extra_fields(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should return True when extra fields are present beyond required."""
        result = populated_registry.validate_payload(
            "test.event",
            {"field_a": "value", "session_id": "sess-123", "extra": "field"},
        )
        assert result is True

    def test_raises_when_required_field_missing(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should raise OnexError when required field is missing."""
        with pytest.raises(
            OnexError,
            match=r"Missing required fields for 'test\.event': \['field_a'\]",
        ):
            populated_registry.validate_payload(
                "test.event", {"session_id": "sess-123"}
            )

    def test_raises_with_all_missing_fields_listed(self) -> None:
        """Should list all missing required fields in error message."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="multi.required",
                topic_template="onex.evt.multi.v1",
                required_fields=("field_a", "field_b", "field_c"),
            )
        )
        with pytest.raises(OnexError) as exc_info:
            registry.validate_payload("multi.required", {"field_a": "value"})
        error_message = str(exc_info.value)
        assert "field_b" in error_message
        assert "field_c" in error_message

    def test_raises_for_unknown_event_type(self) -> None:
        """Should raise OnexError for unknown event type."""
        registry = EventRegistry()
        with pytest.raises(OnexError, match="Unknown event type"):
            registry.validate_payload("unknown.event", {"data": "value"})

    def test_returns_true_when_no_required_fields(self) -> None:
        """Should return True when event type has no required fields."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="optional.all",
                topic_template="onex.evt.optional.v1",
                required_fields=(),
            )
        )
        result = registry.validate_payload("optional.all", {})
        assert result is True


@pytest.mark.unit
class TestEventRegistryInjectMetadata:
    """Tests for inject_metadata() method with deterministic mocking."""

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_adds_correlation_id_when_not_provided(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should generate correlation_id when not provided."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        enriched = populated_registry.inject_metadata(
            "test.event",
            {"field_a": "value"},
        )

        assert enriched["correlation_id"] == str(FIXED_UUID)

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_uses_provided_correlation_id(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should use provided correlation_id instead of generating."""
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        enriched = populated_registry.inject_metadata(
            "test.event",
            {"field_a": "value"},
            correlation_id="custom-corr-id",
        )

        assert enriched["correlation_id"] == "custom-corr-id"
        mock_uuid4.assert_not_called()  # type: ignore[attr-defined]

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_adds_causation_id_when_provided(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should include causation_id when provided."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        enriched = populated_registry.inject_metadata(
            "test.event",
            {"field_a": "value"},
            causation_id="cause-456",
        )

        assert enriched["causation_id"] == "cause-456"

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_causation_id_is_none_when_not_provided(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should set causation_id to None when not provided."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        enriched = populated_registry.inject_metadata(
            "test.event",
            {"field_a": "value"},
        )

        assert enriched["causation_id"] is None

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_adds_emitted_at_timestamp(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should add emitted_at with ISO format timestamp."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        enriched = populated_registry.inject_metadata(
            "test.event",
            {"field_a": "value"},
        )

        assert enriched["emitted_at"] == FIXED_ISO_TIMESTAMP

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_adds_schema_version_from_registration(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should add schema_version from registration."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="versioned.event",
                topic_template="onex.evt.versioned.v2",
                schema_version="2.5.0",
            )
        )

        enriched = registry.inject_metadata(
            "versioned.event",
            {"data": "value"},
        )

        assert enriched["schema_version"] == "2.5.0"

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_default_schema_version_is_1_0_0(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should use default schema_version of 1.0.0."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        enriched = populated_registry.inject_metadata(
            "test.event",
            {"field_a": "value"},
        )

        assert enriched["schema_version"] == "1.0.0"

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_preserves_original_payload_fields(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should preserve all original payload fields."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        original_payload = {
            "field_a": "value",
            "session_id": "sess-123",
            "custom_field": {"nested": "value"},
        }
        enriched = populated_registry.inject_metadata(
            "test.event",
            original_payload,
        )

        assert enriched["field_a"] == "value"
        assert enriched["session_id"] == "sess-123"
        assert enriched["custom_field"] == {"nested": "value"}

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_does_not_modify_original_payload(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should not modify the original payload dictionary."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        original_payload = {"field_a": "value"}
        _ = populated_registry.inject_metadata("test.event", original_payload)

        # Original payload should be unchanged
        assert "correlation_id" not in original_payload
        assert "emitted_at" not in original_payload

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_overwrites_existing_metadata_fields(
        self,
        mock_datetime: object,
        mock_uuid4: object,
        populated_registry: EventRegistry,
    ) -> None:
        """Should overwrite existing metadata fields in payload.

        Note: Based on the implementation, inject_metadata DOES overwrite
        existing correlation_id, causation_id, emitted_at, and schema_version
        fields in the payload. This is the expected behavior as the registry
        is the authoritative source for these fields.
        """
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        payload_with_existing = {
            "field_a": "value",
            "correlation_id": "old-corr-id",
            "emitted_at": "old-timestamp",
            "schema_version": "0.0.1",
        }

        enriched = populated_registry.inject_metadata(
            "test.event",
            payload_with_existing,
        )

        # Registry-injected values should overwrite
        assert enriched["correlation_id"] == str(FIXED_UUID)
        assert enriched["emitted_at"] == FIXED_ISO_TIMESTAMP
        assert enriched["schema_version"] == "1.0.0"

    def test_raises_for_unknown_event_type(self) -> None:
        """Should raise OnexError for unknown event type."""
        registry = EventRegistry()
        with pytest.raises(OnexError, match="Unknown event type"):
            registry.inject_metadata("unknown.event", {"data": "value"})


@pytest.mark.unit
class TestEventRegistryRealmAgnostic:
    """Tests for realm-agnostic topic resolution."""

    def test_dev_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic regardless of dev environment."""
        registry = EventRegistry(environment="dev")
        registry.register(SAMPLE_REGISTRATION)
        topic = registry.resolve_topic("test.event")
        assert topic == "onex.evt.test.topic.v1"
        assert not topic.startswith("dev.")

    def test_prod_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic regardless of prod environment."""
        registry = EventRegistry(environment="prod")
        registry.register(SAMPLE_REGISTRATION)
        topic = registry.resolve_topic("test.event")
        assert topic == "onex.evt.test.topic.v1"
        assert not topic.startswith("prod.")

    def test_staging_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic regardless of staging environment."""
        registry = EventRegistry(environment="staging")
        registry.register(SAMPLE_REGISTRATION)
        topic = registry.resolve_topic("test.event")
        assert topic == "onex.evt.test.topic.v1"
        assert not topic.startswith("staging.")

    def test_custom_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic regardless of custom environment."""
        registry = EventRegistry(environment="my-custom-env")
        registry.register(SAMPLE_REGISTRATION)
        topic = registry.resolve_topic("test.event")
        assert topic == "onex.evt.test.topic.v1"
        assert not topic.startswith("my-custom-env.")

    def test_default_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic even with default environment."""
        registry = EventRegistry()
        registry.register(SAMPLE_REGISTRATION)
        topic = registry.resolve_topic("test.event")
        assert topic == "onex.evt.test.topic.v1"
        assert not topic.startswith("dev.")


@pytest.mark.unit
class TestEventRegistryListEventTypes:
    """Tests for list_event_types() method."""

    def test_returns_empty_list_for_fresh_registry(self) -> None:
        """Should return empty list for a fresh registry."""
        registry = EventRegistry()
        assert registry.list_event_types() == []

    def test_includes_registered_types(self, populated_registry: EventRegistry) -> None:
        """Should include registered event types in list."""
        event_types = populated_registry.list_event_types()
        assert "test.event" in event_types

    def test_includes_custom_registrations(self) -> None:
        """Should include custom registrations in list."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="custom.event",
                topic_template="onex.evt.custom.v1",
            )
        )
        event_types = registry.list_event_types()
        assert "custom.event" in event_types

    def test_returns_list_type(self) -> None:
        """Should return a list, not other iterable type."""
        registry = EventRegistry()
        event_types = registry.list_event_types()
        assert isinstance(event_types, list)

    def test_count_increases_with_registrations(self) -> None:
        """Should reflect new registrations in count."""
        registry = EventRegistry()
        assert len(registry.list_event_types()) == 0

        registry.register(
            ModelEventRegistration(
                event_type="new.event.one",
                topic_template="onex.evt.new.one.v1",
            )
        )
        registry.register(
            ModelEventRegistration(
                event_type="new.event.two",
                topic_template="onex.evt.new.two.v1",
            )
        )

        assert len(registry.list_event_types()) == 2


@pytest.mark.unit
class TestEventRegistryGetRegistration:
    """Tests for get_registration() method."""

    def test_returns_registration_for_known_type(
        self, populated_registry: EventRegistry
    ) -> None:
        """Should return registration for known event type."""
        registration = populated_registry.get_registration("test.event")
        assert registration is not None
        assert registration.event_type == "test.event"
        assert registration.partition_key_field == "session_id"

    def test_returns_none_for_unknown_type(self) -> None:
        """Should return None for unknown event type."""
        registry = EventRegistry()
        registration = registry.get_registration("unknown.event")
        assert registration is None

    def test_returns_custom_registration(self) -> None:
        """Should return custom registration after register()."""
        registry = EventRegistry()
        custom = ModelEventRegistration(
            event_type="custom.event",
            topic_template="onex.evt.custom.v1",
            partition_key_field="custom_key",
            required_fields=("a", "b"),
            schema_version="3.0.0",
        )
        registry.register(custom)

        retrieved = registry.get_registration("custom.event")
        assert retrieved is not None
        assert retrieved.partition_key_field == "custom_key"
        assert retrieved.required_fields == ("a", "b")
        assert retrieved.schema_version == "3.0.0"


@pytest.mark.unit
class TestPhaseMetricsRegistration:
    """Tests for the pre-built PHASE_METRICS_REGISTRATION constant."""

    def test_phase_metrics_event_type(self) -> None:
        """Should have correct event type."""
        assert PHASE_METRICS_REGISTRATION.event_type == "phase.metrics"

    def test_phase_metrics_topic_template(self) -> None:
        """Should map to correct Kafka topic."""
        assert (
            PHASE_METRICS_REGISTRATION.topic_template
            == "onex.evt.omniclaude.phase-metrics.v1"
        )

    def test_phase_metrics_partition_key(self) -> None:
        """Should partition by run_id."""
        assert PHASE_METRICS_REGISTRATION.partition_key_field == "run_id"

    def test_phase_metrics_resolves_in_registry(self) -> None:
        """Should resolve correctly when registered in EventRegistry."""
        registry = EventRegistry()
        registry.register(PHASE_METRICS_REGISTRATION)
        topic = registry.resolve_topic("phase.metrics")
        assert topic == "onex.evt.omniclaude.phase-metrics.v1"

    def test_phase_metrics_partition_key_extraction(self) -> None:
        """Should extract run_id as partition key from payload."""
        registry = EventRegistry()
        registry.register(PHASE_METRICS_REGISTRATION)
        key = registry.get_partition_key("phase.metrics", {"run_id": "run-abc-123"})
        assert key == "run-abc-123"
