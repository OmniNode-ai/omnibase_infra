#!/usr/bin/env python3
"""Unit tests for EnumReducerEvent.

Tests cover:
- Enum value definitions
- Kafka topic name generation
- Terminal event detection
- Event type validation
- String representation
"""

import pytest

from omninode_bridge.nodes.reducer.v1_0_0.models.enum_reducer_event import (
    EnumReducerEvent,
)


class TestEnumReducerEvent:
    """Test suite for EnumReducerEvent."""

    def test_enum_values(self):
        """Test that all expected enum values are defined."""
        assert EnumReducerEvent.AGGREGATION_STARTED.value == "aggregation_started"
        assert EnumReducerEvent.BATCH_PROCESSED.value == "batch_processed"
        assert EnumReducerEvent.STATE_PERSISTED.value == "state_persisted"
        assert EnumReducerEvent.AGGREGATION_COMPLETED.value == "aggregation_completed"
        assert EnumReducerEvent.AGGREGATION_FAILED.value == "aggregation_failed"
        assert EnumReducerEvent.FSM_STATE_INITIALIZED.value == "fsm_state_initialized"
        assert EnumReducerEvent.FSM_STATE_TRANSITIONED.value == "fsm_state_transitioned"
        assert EnumReducerEvent.NODE_INTROSPECTION.value == "node_introspection"
        assert (
            EnumReducerEvent.REGISTRY_REQUEST_INTROSPECTION.value
            == "registry_request_introspection"
        )
        assert EnumReducerEvent.NODE_HEARTBEAT.value == "node_heartbeat"

    def test_enum_inheritance(self):
        """Test enum inherits from str."""
        assert isinstance(EnumReducerEvent.AGGREGATION_STARTED, str)

    def test_enum_string_behavior(self):
        """Test enum behaves like string."""
        assert EnumReducerEvent.AGGREGATION_STARTED.value == "aggregation_started"
        assert EnumReducerEvent.AGGREGATION_STARTED == "aggregation_started"

    def test_enum_iteration(self):
        """Test enum iteration."""
        all_events = list(EnumReducerEvent)
        assert len(all_events) == 10  # Total number of events
        assert EnumReducerEvent.AGGREGATION_STARTED in all_events
        assert EnumReducerEvent.AGGREGATION_COMPLETED in all_events

    def test_enum_membership(self):
        """Test enum membership."""
        assert EnumReducerEvent.AGGREGATION_STARTED == "aggregation_started"
        assert EnumReducerEvent.AGGREGATION_STARTED != "invalid_event"

    def test_get_topic_name_default_namespace(self):
        """Test Kafka topic name generation with default namespace."""
        topic = EnumReducerEvent.AGGREGATION_STARTED.get_topic_name()
        assert topic == "dev.omninode_bridge.onex.evt.aggregation-started.v1"

    def test_get_topic_name_custom_namespace(self):
        """Test Kafka topic name generation with custom namespace."""
        topic = EnumReducerEvent.BATCH_PROCESSED.get_topic_name(namespace="prod")
        assert topic == "prod.omninode_bridge.onex.evt.batch-processed.v1"

    def test_get_topic_name_all_events(self):
        """Test topic name generation for all events."""
        # Test that all events can generate topic names
        for event in EnumReducerEvent:
            topic = event.get_topic_name()
            assert topic.startswith("dev.omninode_bridge.onex.evt.")
            assert topic.endswith(".v1")
            # Verify event value is converted (underscores → hyphens)
            event_slug = event.value.replace("_", "-")
            assert event_slug in topic

    def test_get_topic_name_underscore_conversion(self):
        """Test that underscores in event values are converted to hyphens in topic names."""
        topic = EnumReducerEvent.FSM_STATE_INITIALIZED.get_topic_name()
        # fsm_state_initialized → fsm-state-initialized
        assert "fsm-state-initialized" in topic
        assert "fsm_state_initialized" not in topic

    def test_is_terminal_event(self):
        """Test terminal event detection."""
        # Terminal events
        assert EnumReducerEvent.AGGREGATION_COMPLETED.is_terminal_event is True
        assert EnumReducerEvent.AGGREGATION_FAILED.is_terminal_event is True

        # Non-terminal events
        assert EnumReducerEvent.AGGREGATION_STARTED.is_terminal_event is False
        assert EnumReducerEvent.BATCH_PROCESSED.is_terminal_event is False
        assert EnumReducerEvent.STATE_PERSISTED.is_terminal_event is False
        assert EnumReducerEvent.FSM_STATE_INITIALIZED.is_terminal_event is False
        assert EnumReducerEvent.FSM_STATE_TRANSITIONED.is_terminal_event is False
        assert EnumReducerEvent.NODE_INTROSPECTION.is_terminal_event is False
        assert (
            EnumReducerEvent.REGISTRY_REQUEST_INTROSPECTION.is_terminal_event is False
        )
        assert EnumReducerEvent.NODE_HEARTBEAT.is_terminal_event is False

    def test_enum_docstring(self):
        """Test enum has docstring."""
        assert EnumReducerEvent.__doc__ is not None
        assert "Kafka event types" in EnumReducerEvent.__doc__

    def test_enum_value_docstrings(self):
        """Test enum values have docstrings."""
        # Check a few key events have docstrings
        assert EnumReducerEvent.AGGREGATION_STARTED.__doc__ is not None
        assert EnumReducerEvent.AGGREGATION_COMPLETED.__doc__ is not None
        assert EnumReducerEvent.FSM_STATE_INITIALIZED.__doc__ is not None

    def test_enum_serialization(self):
        """Test enum serialization to string."""
        event = EnumReducerEvent.AGGREGATION_STARTED
        # Can be serialized to string via .value
        serialized = event.value
        assert serialized == "aggregation_started"

        # Can be compared with string
        assert event == serialized

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        event1 = EnumReducerEvent.AGGREGATION_STARTED
        event2 = EnumReducerEvent.AGGREGATION_STARTED
        event3 = EnumReducerEvent.BATCH_PROCESSED

        # Same events are equal
        assert event1 == event2
        assert event1 is event2  # Same instance

        # Different events are not equal
        assert event1 != event3

    def test_enum_invalid_value(self):
        """Test accessing invalid enum value raises error."""
        with pytest.raises(ValueError):
            EnumReducerEvent("invalid_event_name")
