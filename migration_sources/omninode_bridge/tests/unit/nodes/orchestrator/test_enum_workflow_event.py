#!/usr/bin/env python3
"""Unit tests for EnumWorkflowEvent.

Tests cover:
- Enum values and inheritance
- String behavior and serialization
- Topic name generation
- Terminal event detection
- All enum values and properties
"""

import json
from enum import Enum

import pytest

from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_event import (
    EnumWorkflowEvent,
)


class TestEnumWorkflowEvent:
    """Test suite for EnumWorkflowEvent."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert EnumWorkflowEvent.WORKFLOW_STARTED == "stamp_workflow_started"
        assert EnumWorkflowEvent.WORKFLOW_COMPLETED == "stamp_workflow_completed"
        assert EnumWorkflowEvent.WORKFLOW_FAILED == "stamp_workflow_failed"
        assert EnumWorkflowEvent.STEP_COMPLETED == "workflow_step_completed"
        assert (
            EnumWorkflowEvent.INTELLIGENCE_REQUESTED
            == "onextree_intelligence_requested"
        )
        assert (
            EnumWorkflowEvent.INTELLIGENCE_RECEIVED == "onextree_intelligence_received"
        )
        assert EnumWorkflowEvent.STAMP_CREATED == "metadata_stamp_created"
        assert EnumWorkflowEvent.HASH_GENERATED == "blake3_hash_generated"
        assert EnumWorkflowEvent.STATE_TRANSITION == "workflow_state_transition"
        assert EnumWorkflowEvent.NODE_INTROSPECTION == "node_introspection"
        assert (
            EnumWorkflowEvent.REGISTRY_REQUEST_INTROSPECTION
            == "registry_request_introspection"
        )
        assert EnumWorkflowEvent.NODE_HEARTBEAT == "node_heartbeat"

    def test_enum_inheritance(self):
        """Test that enum inherits from str and Enum."""
        assert issubclass(EnumWorkflowEvent, str)
        assert issubclass(EnumWorkflowEvent, Enum)

    def test_enum_string_behavior(self):
        """Test that enum behaves like a string."""
        event = EnumWorkflowEvent.WORKFLOW_STARTED
        assert isinstance(event, str)
        assert event == "stamp_workflow_started"
        assert len(event) == 22

    def test_enum_serialization(self):
        """Test enum serialization to JSON."""
        event = EnumWorkflowEvent.WORKFLOW_COMPLETED
        serialized = event.value
        assert serialized == "stamp_workflow_completed"

        # Test JSON serialization
        json_str = json.dumps(event)
        assert json_str == '"stamp_workflow_completed"'

        # Test JSON deserialization
        deserialized = json.loads(json_str)
        assert deserialized == "stamp_workflow_completed"

    def test_enum_iteration(self):
        """Test that all enum values can be iterated."""
        values = list(EnumWorkflowEvent)
        assert len(values) == 12  # Verify we have all expected values

        # Check that specific values are in the iteration
        assert EnumWorkflowEvent.WORKFLOW_STARTED in values
        assert EnumWorkflowEvent.WORKFLOW_COMPLETED in values
        assert EnumWorkflowEvent.WORKFLOW_FAILED in values
        assert EnumWorkflowEvent.STEP_COMPLETED in values
        assert EnumWorkflowEvent.INTELLIGENCE_REQUESTED in values
        assert EnumWorkflowEvent.INTELLIGENCE_RECEIVED in values
        assert EnumWorkflowEvent.STAMP_CREATED in values
        assert EnumWorkflowEvent.HASH_GENERATED in values
        assert EnumWorkflowEvent.STATE_TRANSITION in values
        assert EnumWorkflowEvent.NODE_INTROSPECTION in values
        assert EnumWorkflowEvent.REGISTRY_REQUEST_INTROSPECTION in values
        assert EnumWorkflowEvent.NODE_HEARTBEAT in values

    def test_enum_membership(self):
        """Test enum membership operations."""
        assert "stamp_workflow_started" in EnumWorkflowEvent
        assert "stamp_workflow_completed" in EnumWorkflowEvent
        assert "stamp_workflow_failed" in EnumWorkflowEvent
        assert "workflow_step_completed" in EnumWorkflowEvent
        assert "onextree_intelligence_requested" in EnumWorkflowEvent
        assert "onextree_intelligence_received" in EnumWorkflowEvent
        assert "metadata_stamp_created" in EnumWorkflowEvent
        assert "blake3_hash_generated" in EnumWorkflowEvent
        assert "workflow_state_transition" in EnumWorkflowEvent
        assert "node_introspection" in EnumWorkflowEvent
        assert "registry_request_introspection" in EnumWorkflowEvent
        assert "node_heartbeat" in EnumWorkflowEvent

        assert "invalid_event" not in EnumWorkflowEvent

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        event1 = EnumWorkflowEvent.WORKFLOW_STARTED
        event2 = EnumWorkflowEvent.WORKFLOW_STARTED
        assert event1 == event2
        assert event1 is event2

        # Test string comparison
        assert event1 == "stamp_workflow_started"
        assert event1 == "stamp_workflow_started"

    def test_enum_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            EnumWorkflowEvent("invalid_event")

    def test_get_topic_name_default_namespace(self):
        """Test topic name generation with default namespace."""
        event = EnumWorkflowEvent.WORKFLOW_STARTED
        topic = event.get_topic_name()
        assert topic == "dev.omninode_bridge.onex.evt.stamp-workflow-started.v1"

        event = EnumWorkflowEvent.WORKFLOW_COMPLETED
        topic = event.get_topic_name()
        assert topic == "dev.omninode_bridge.onex.evt.stamp-workflow-completed.v1"

        event = EnumWorkflowEvent.INTELLIGENCE_REQUESTED
        topic = event.get_topic_name()
        assert (
            topic == "dev.omninode_bridge.onex.evt.onextree-intelligence-requested.v1"
        )

    def test_get_topic_name_custom_namespace(self):
        """Test topic name generation with custom namespace."""
        event = EnumWorkflowEvent.WORKFLOW_STARTED
        topic = event.get_topic_name("production")
        assert topic == "production.omninode_bridge.onex.evt.stamp-workflow-started.v1"

        event = EnumWorkflowEvent.NODE_HEARTBEAT
        topic = event.get_topic_name("test.namespace")
        assert topic == "test.namespace.omninode_bridge.onex.evt.node-heartbeat.v1"

    def test_get_topic_name_all_events(self):
        """Test topic name generation for all events."""
        expected_topics = {
            EnumWorkflowEvent.WORKFLOW_STARTED: "dev.omninode_bridge.onex.evt.stamp-workflow-started.v1",
            EnumWorkflowEvent.WORKFLOW_COMPLETED: "dev.omninode_bridge.onex.evt.stamp-workflow-completed.v1",
            EnumWorkflowEvent.WORKFLOW_FAILED: "dev.omninode_bridge.onex.evt.stamp-workflow-failed.v1",
            EnumWorkflowEvent.STEP_COMPLETED: "dev.omninode_bridge.onex.evt.workflow-step-completed.v1",
            EnumWorkflowEvent.INTELLIGENCE_REQUESTED: "dev.omninode_bridge.onex.evt.onextree-intelligence-requested.v1",
            EnumWorkflowEvent.INTELLIGENCE_RECEIVED: "dev.omninode_bridge.onex.evt.onextree-intelligence-received.v1",
            EnumWorkflowEvent.STAMP_CREATED: "dev.omninode_bridge.onex.evt.metadata-stamp-created.v1",
            EnumWorkflowEvent.HASH_GENERATED: "dev.omninode_bridge.onex.evt.blake3-hash-generated.v1",
            EnumWorkflowEvent.STATE_TRANSITION: "dev.omninode_bridge.onex.evt.workflow-state-transition.v1",
            EnumWorkflowEvent.NODE_INTROSPECTION: "dev.omninode_bridge.onex.evt.node-introspection.v1",
            EnumWorkflowEvent.REGISTRY_REQUEST_INTROSPECTION: "dev.omninode_bridge.onex.evt.registry-request-introspection.v1",
            EnumWorkflowEvent.NODE_HEARTBEAT: "dev.omninode_bridge.onex.evt.node-heartbeat.v1",
        }

        for event, expected_topic in expected_topics.items():
            assert event.get_topic_name() == expected_topic

    def test_is_terminal_event(self):
        """Test terminal event detection."""
        # Terminal events
        assert EnumWorkflowEvent.WORKFLOW_COMPLETED.is_terminal_event is True
        assert EnumWorkflowEvent.WORKFLOW_FAILED.is_terminal_event is True

        # Non-terminal events
        assert EnumWorkflowEvent.WORKFLOW_STARTED.is_terminal_event is False
        assert EnumWorkflowEvent.STEP_COMPLETED.is_terminal_event is False
        assert EnumWorkflowEvent.INTELLIGENCE_REQUESTED.is_terminal_event is False
        assert EnumWorkflowEvent.INTELLIGENCE_RECEIVED.is_terminal_event is False
        assert EnumWorkflowEvent.STAMP_CREATED.is_terminal_event is False
        assert EnumWorkflowEvent.HASH_GENERATED.is_terminal_event is False
        assert EnumWorkflowEvent.STATE_TRANSITION.is_terminal_event is False
        assert EnumWorkflowEvent.NODE_INTROSPECTION.is_terminal_event is False
        assert (
            EnumWorkflowEvent.REGISTRY_REQUEST_INTROSPECTION.is_terminal_event is False
        )
        assert EnumWorkflowEvent.NODE_HEARTBEAT.is_terminal_event is False

    def test_enum_docstring(self):
        """Test that enum has proper docstring."""
        assert EnumWorkflowEvent.__doc__ is not None
        assert "Kafka event types" in EnumWorkflowEvent.__doc__

    def test_enum_value_docstrings(self):
        """Test that enum class has comprehensive documentation.

        Note: Python enums expose the class docstring via __doc__, not individual
        value docstrings. The value docstrings exist in source for developer reference.
        """
        # Class docstring should describe all events
        class_doc = EnumWorkflowEvent.__doc__
        assert class_doc is not None
        assert "WORKFLOW_STARTED" in class_doc
        assert "WORKFLOW_COMPLETED" in class_doc
        assert "WORKFLOW_FAILED" in class_doc
        assert "STEP_COMPLETED" in class_doc
        assert "INTELLIGENCE_REQUESTED" in class_doc
