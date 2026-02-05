# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EventRegistry - Event type to Kafka topic mapping.

Tests the event registry that maps semantic event types (e.g., "prompt.submitted")
to Kafka topics and handles metadata injection for the Hook Event Daemon (OMN-1610).

Test coverage includes:
- Default event type registrations
- Topic resolution with environment prefixes
- Custom event registration and overrides
- Partition key extraction
- Payload validation
- Metadata injection with deterministic mocking
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_core.errors import OnexError
from omnibase_infra.runtime.emit_daemon.event_registry import (
    EventRegistry,
    ModelEventRegistration,
)

# Fixed values for deterministic tests
FIXED_UUID = UUID("12345678-1234-5678-1234-567812345678")
FIXED_TIMESTAMP = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
FIXED_ISO_TIMESTAMP = "2025-01-15T10:30:00+00:00"


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
        assert reg.required_fields == []
        assert reg.schema_version == "1.0.0"

    def test_full_registration(self) -> None:
        """Should create registration with all fields specified."""
        reg = ModelEventRegistration(
            event_type="custom.event",
            topic_template="onex.evt.custom.topic.v2",
            partition_key_field="user_id",
            required_fields=["user_id", "action"],
            schema_version="2.0.0",
        )
        assert reg.event_type == "custom.event"
        assert reg.topic_template == "onex.evt.custom.topic.v2"
        assert reg.partition_key_field == "user_id"
        assert reg.required_fields == ["user_id", "action"]
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


class TestEventRegistryDefaultRegistrations:
    """Tests for default event type registrations."""

    def test_prompt_submitted_default(self) -> None:
        """Should register prompt.submitted with correct topic template."""
        registry = EventRegistry(environment="dev")
        topic = registry.resolve_topic("prompt.submitted")
        assert topic == "onex.evt.omniclaude.prompt-submitted.v1"

    def test_session_started_default(self) -> None:
        """Should register session.started with correct topic template."""
        registry = EventRegistry(environment="dev")
        topic = registry.resolve_topic("session.started")
        assert topic == "onex.evt.omniclaude.session-started.v1"

    def test_session_ended_default(self) -> None:
        """Should register session.ended with correct topic template."""
        registry = EventRegistry(environment="dev")
        topic = registry.resolve_topic("session.ended")
        assert topic == "onex.evt.omniclaude.session-ended.v1"

    def test_tool_executed_default(self) -> None:
        """Should register tool.executed with correct topic template."""
        registry = EventRegistry(environment="dev")
        topic = registry.resolve_topic("tool.executed")
        assert topic == "onex.evt.omniclaude.tool-executed.v1"

    def test_all_defaults_registered(self) -> None:
        """Should register all default event types."""
        registry = EventRegistry()
        event_types = registry.list_event_types()
        expected = [
            "prompt.submitted",
            "session.started",
            "session.ended",
            "session.outcome",
            "tool.executed",
            "injection.recorded",
            "context.utilization",
            "agent.match",
            "latency.breakdown",
            "notification.blocked",
            "notification.completed",
        ]
        assert sorted(event_types) == sorted(expected)


class TestEventRegistryResolveTopic:
    """Tests for resolve_topic() method."""

    def test_resolve_topic_with_environment_prefix(self) -> None:
        """Should return correct topic with environment prefix."""
        registry = EventRegistry(environment="staging")
        topic = registry.resolve_topic("prompt.submitted")
        assert topic == "onex.evt.omniclaude.prompt-submitted.v1"

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

    def test_resolve_topic_error_includes_registered_types(self) -> None:
        """Should include registered types in error message."""
        registry = EventRegistry()
        with pytest.raises(OnexError) as exc_info:
            registry.resolve_topic("unknown.event")
        error_message = str(exc_info.value)
        assert "prompt.submitted" in error_message
        assert "session.started" in error_message


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

    def test_register_can_override_default(self) -> None:
        """Should allow overriding default registrations."""
        registry = EventRegistry(environment="dev")

        # Override prompt.submitted with custom topic
        registry.register(
            ModelEventRegistration(
                event_type="prompt.submitted",
                topic_template="onex.evt.custom.prompt.v2",
            )
        )

        topic = registry.resolve_topic("prompt.submitted")
        assert topic == "onex.evt.custom.prompt.v2"

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


class TestEventRegistryGetPartitionKey:
    """Tests for get_partition_key() method."""

    def test_extracts_partition_key_from_payload(self) -> None:
        """Should extract partition key based on configured field."""
        registry = EventRegistry()
        # prompt.submitted has partition_key_field="session_id"
        key = registry.get_partition_key(
            "prompt.submitted",
            {"prompt": "Hello", "session_id": "sess-abc123"},
        )
        assert key == "sess-abc123"

    def test_returns_none_when_field_not_in_payload(self) -> None:
        """Should return None when partition key field is missing from payload."""
        registry = EventRegistry()
        key = registry.get_partition_key(
            "prompt.submitted",
            {"prompt": "Hello"},  # No session_id
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

    def test_partition_key_returns_none_for_none_value(self) -> None:
        """Should return None when partition key field value is None."""
        registry = EventRegistry()
        key = registry.get_partition_key(
            "prompt.submitted",
            {"prompt": "Hello", "session_id": None},
        )
        assert key is None


class TestEventRegistryValidatePayload:
    """Tests for validate_payload() method."""

    def test_returns_true_when_all_required_fields_present(self) -> None:
        """Should return True when all required fields are present."""
        registry = EventRegistry()
        # prompt.submitted requires ["prompt"]
        result = registry.validate_payload(
            "prompt.submitted",
            {"prompt": "Hello, world!"},
        )
        assert result is True

    def test_returns_true_with_extra_fields(self) -> None:
        """Should return True when extra fields are present beyond required."""
        registry = EventRegistry()
        result = registry.validate_payload(
            "prompt.submitted",
            {"prompt": "Hello", "session_id": "sess-123", "extra": "field"},
        )
        assert result is True

    def test_raises_when_required_field_missing(self) -> None:
        """Should raise OnexError when required field is missing."""
        registry = EventRegistry()
        with pytest.raises(
            OnexError,
            match=r"Missing required fields for 'prompt\.submitted': \['prompt'\]",
        ):
            registry.validate_payload("prompt.submitted", {"session_id": "sess-123"})

    def test_raises_with_all_missing_fields_listed(self) -> None:
        """Should list all missing required fields in error message."""
        registry = EventRegistry()
        registry.register(
            ModelEventRegistration(
                event_type="multi.required",
                topic_template="onex.evt.multi.v1",
                required_fields=["field_a", "field_b", "field_c"],
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
                required_fields=[],
            )
        )
        result = registry.validate_payload("optional.all", {})
        assert result is True


class TestEventRegistryInjectMetadata:
    """Tests for inject_metadata() method with deterministic mocking."""

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_adds_correlation_id_when_not_provided(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should generate correlation_id when not provided."""
        # Configure mocks using setattr to avoid type issues
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        enriched = registry.inject_metadata(
            "prompt.submitted",
            {"prompt": "Hello"},
        )

        assert enriched["correlation_id"] == str(FIXED_UUID)

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_uses_provided_correlation_id(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should use provided correlation_id instead of generating."""
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        enriched = registry.inject_metadata(
            "prompt.submitted",
            {"prompt": "Hello"},
            correlation_id="custom-corr-id",
        )

        assert enriched["correlation_id"] == "custom-corr-id"
        mock_uuid4.assert_not_called()  # type: ignore[attr-defined]

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_adds_causation_id_when_provided(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should include causation_id when provided."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        enriched = registry.inject_metadata(
            "prompt.submitted",
            {"prompt": "Hello"},
            causation_id="cause-456",
        )

        assert enriched["causation_id"] == "cause-456"

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_causation_id_is_none_when_not_provided(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should set causation_id to None when not provided."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        enriched = registry.inject_metadata(
            "prompt.submitted",
            {"prompt": "Hello"},
        )

        assert enriched["causation_id"] is None

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_adds_emitted_at_timestamp(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should add emitted_at with ISO format timestamp."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        enriched = registry.inject_metadata(
            "prompt.submitted",
            {"prompt": "Hello"},
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
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should use default schema_version of 1.0.0."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        enriched = registry.inject_metadata(
            "prompt.submitted",
            {"prompt": "Hello"},
        )

        assert enriched["schema_version"] == "1.0.0"

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_preserves_original_payload_fields(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should preserve all original payload fields."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        original_payload = {
            "prompt": "Hello, world!",
            "session_id": "sess-123",
            "custom_field": {"nested": "value"},
        }
        enriched = registry.inject_metadata(
            "prompt.submitted",
            original_payload,
        )

        assert enriched["prompt"] == "Hello, world!"
        assert enriched["session_id"] == "sess-123"
        assert enriched["custom_field"] == {"nested": "value"}

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_does_not_modify_original_payload(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should not modify the original payload dictionary."""
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        original_payload = {"prompt": "Hello"}
        _ = registry.inject_metadata("prompt.submitted", original_payload)

        # Original payload should be unchanged
        assert "correlation_id" not in original_payload
        assert "emitted_at" not in original_payload

    @patch("omnibase_infra.runtime.emit_daemon.event_registry.uuid4")
    @patch("omnibase_infra.runtime.emit_daemon.event_registry.datetime")
    def test_overwrites_existing_metadata_fields(
        self, mock_datetime: object, mock_uuid4: object
    ) -> None:
        """Should overwrite existing metadata fields in payload.

        Note: Based on the implementation, inject_metadata DOES overwrite
        existing correlation_id, causation_id, emitted_at, and schema_version
        fields in the payload. This is the expected behavior as the registry
        is the authoritative source for these fields.
        """
        mock_uuid4.return_value = FIXED_UUID  # type: ignore[attr-defined]
        mock_datetime.now.return_value = FIXED_TIMESTAMP  # type: ignore[attr-defined]

        registry = EventRegistry()
        payload_with_existing = {
            "prompt": "Hello",
            "correlation_id": "old-corr-id",
            "emitted_at": "old-timestamp",
            "schema_version": "0.0.1",
        }

        enriched = registry.inject_metadata(
            "prompt.submitted",
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


class TestEventRegistryRealmAgnostic:
    """Tests for realm-agnostic topic resolution."""

    def test_dev_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic regardless of dev environment."""
        registry = EventRegistry(environment="dev")
        topic = registry.resolve_topic("prompt.submitted")
        assert topic == "onex.evt.omniclaude.prompt-submitted.v1"
        assert not topic.startswith("dev.")

    def test_prod_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic regardless of prod environment."""
        registry = EventRegistry(environment="prod")
        topic = registry.resolve_topic("prompt.submitted")
        assert topic == "onex.evt.omniclaude.prompt-submitted.v1"
        assert not topic.startswith("prod.")

    def test_staging_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic regardless of staging environment."""
        registry = EventRegistry(environment="staging")
        topic = registry.resolve_topic("prompt.submitted")
        assert topic == "onex.evt.omniclaude.prompt-submitted.v1"
        assert not topic.startswith("staging.")

    def test_custom_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic regardless of custom environment."""
        registry = EventRegistry(environment="my-custom-env")
        topic = registry.resolve_topic("prompt.submitted")
        assert topic == "onex.evt.omniclaude.prompt-submitted.v1"
        assert not topic.startswith("my-custom-env.")

    def test_default_environment_topic_is_realm_agnostic(self) -> None:
        """Topic should be realm-agnostic even with default environment."""
        registry = EventRegistry()
        topic = registry.resolve_topic("prompt.submitted")
        assert topic == "onex.evt.omniclaude.prompt-submitted.v1"
        assert not topic.startswith("dev.")


class TestEventRegistryListEventTypes:
    """Tests for list_event_types() method."""

    def test_returns_all_default_event_types(self) -> None:
        """Should return all default registered event types."""
        registry = EventRegistry()
        event_types = registry.list_event_types()
        assert "prompt.submitted" in event_types
        assert "session.started" in event_types
        assert "session.ended" in event_types
        assert "tool.executed" in event_types

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
        initial_count = len(registry.list_event_types())

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

        final_count = len(registry.list_event_types())
        assert final_count == initial_count + 2


class TestEventRegistryGetRegistration:
    """Tests for get_registration() method."""

    def test_returns_registration_for_known_type(self) -> None:
        """Should return registration for known event type."""
        registry = EventRegistry()
        registration = registry.get_registration("prompt.submitted")
        assert registration is not None
        assert registration.event_type == "prompt.submitted"
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
            required_fields=["a", "b"],
            schema_version="3.0.0",
        )
        registry.register(custom)

        retrieved = registry.get_registration("custom.event")
        assert retrieved is not None
        assert retrieved.partition_key_field == "custom_key"
        assert retrieved.required_fields == ["a", "b"]
        assert retrieved.schema_version == "3.0.0"


class TestEventRegistryDefaultRegistrationDetails:
    """Tests verifying specific details of default registrations."""

    def test_prompt_submitted_partition_key(self) -> None:
        """prompt.submitted should use session_id as partition key."""
        registry = EventRegistry()
        reg = registry.get_registration("prompt.submitted")
        assert reg is not None
        assert reg.partition_key_field == "session_id"

    def test_prompt_submitted_required_fields(self) -> None:
        """prompt.submitted should require prompt field."""
        registry = EventRegistry()
        reg = registry.get_registration("prompt.submitted")
        assert reg is not None
        assert "prompt" in reg.required_fields

    def test_session_started_required_fields(self) -> None:
        """session.started should require session_id field."""
        registry = EventRegistry()
        reg = registry.get_registration("session.started")
        assert reg is not None
        assert "session_id" in reg.required_fields

    def test_session_ended_required_fields(self) -> None:
        """session.ended should require session_id field."""
        registry = EventRegistry()
        reg = registry.get_registration("session.ended")
        assert reg is not None
        assert "session_id" in reg.required_fields

    def test_tool_executed_required_fields(self) -> None:
        """tool.executed should require tool_name field."""
        registry = EventRegistry()
        reg = registry.get_registration("tool.executed")
        assert reg is not None
        assert "tool_name" in reg.required_fields

    def test_tool_executed_partition_key(self) -> None:
        """tool.executed should use session_id as partition key."""
        registry = EventRegistry()
        reg = registry.get_registration("tool.executed")
        assert reg is not None
        assert reg.partition_key_field == "session_id"

    def test_notification_blocked_required_fields(self) -> None:
        """notification.blocked should require ticket_id, reason, repo, session_id."""
        registry = EventRegistry()
        reg = registry.get_registration("notification.blocked")
        assert reg is not None
        assert "ticket_id" in reg.required_fields
        assert "reason" in reg.required_fields
        assert "repo" in reg.required_fields
        assert "session_id" in reg.required_fields

    def test_notification_blocked_partition_key(self) -> None:
        """notification.blocked should use session_id as partition key."""
        registry = EventRegistry()
        reg = registry.get_registration("notification.blocked")
        assert reg is not None
        assert reg.partition_key_field == "session_id"

    def test_notification_blocked_topic(self) -> None:
        """notification.blocked should have correct topic template."""
        registry = EventRegistry()
        topic = registry.resolve_topic("notification.blocked")
        assert topic == "onex.evt.omniclaude.notification-blocked.v1"

    def test_notification_completed_required_fields(self) -> None:
        """notification.completed should require ticket_id, summary, repo, session_id."""
        registry = EventRegistry()
        reg = registry.get_registration("notification.completed")
        assert reg is not None
        assert "ticket_id" in reg.required_fields
        assert "summary" in reg.required_fields
        assert "repo" in reg.required_fields
        assert "session_id" in reg.required_fields

    def test_notification_completed_partition_key(self) -> None:
        """notification.completed should use session_id as partition key."""
        registry = EventRegistry()
        reg = registry.get_registration("notification.completed")
        assert reg is not None
        assert reg.partition_key_field == "session_id"

    def test_notification_completed_topic(self) -> None:
        """notification.completed should have correct topic template."""
        registry = EventRegistry()
        topic = registry.resolve_topic("notification.completed")
        assert topic == "onex.evt.omniclaude.notification-completed.v1"
