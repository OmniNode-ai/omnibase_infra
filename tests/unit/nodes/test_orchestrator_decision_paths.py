# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for orchestrator decision path event types (OMN-952).

These tests verify that the Node Registration Orchestrator contract correctly
defines all 7 decision path event types as specified in OMN-888. The orchestrator
emits different events based on workflow decisions:

Decision Path Event Types:
    1. NodeRegistrationInitiated - Emitted when a registration workflow starts.
       Triggered when an introspection event is received for a new node.

    2. NodeRegistrationAccepted - Emitted when registration is successfully accepted.
       Triggered after both Consul and PostgreSQL registrations succeed.

    3. NodeRegistrationRejected - Emitted when registration is rejected.
       Triggered when validation fails or the node is not eligible for registration.

    4. NodeRegistrationAckTimedOut - Emitted when acknowledgment times out.
       Triggered when the node fails to acknowledge within the configured timeout.

    5. NodeRegistrationAckReceived - Emitted when acknowledgment is received.
       Triggered when the node successfully acknowledges its registration.

    6. NodeBecameActive - Emitted when a node transitions to active state.
       Triggered after successful registration and acknowledgment.

    7. NodeLivenessExpired - Emitted when a node's liveness check fails.
       Triggered when heartbeat or health check timeout is exceeded.

Additionally, there is one result event:
    - NodeRegistrationResultEvent - Contains the final outcome of the workflow.

Topic Convention:
    All events follow the pattern: {env}.{namespace}.onex.evt.<event-slug>.v1
    Example: dev.myapp.onex.evt.node-registration-initiated.v1

Running Tests:
    # Run all decision path tests:
    pytest tests/unit/nodes/test_orchestrator_decision_paths.py -v

    # Run specific test:
    pytest tests/unit/nodes/test_orchestrator_decision_paths.py::TestDecisionPathEvents::test_contract_publishes_all_7_event_types -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def contract_path() -> Path:
    """Return path to the orchestrator contract.yaml.

    Raises:
        pytest.skip: If contract file doesn't exist.
    """
    path = Path("src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml")
    if not path.exists():
        pytest.skip(f"Contract file not found: {path}")
    return path


@pytest.fixture
def contract_data(contract_path: Path) -> dict:
    """Load and return contract.yaml as dict.

    Raises:
        pytest.fail: If contract file contains invalid YAML.
    """
    with open(contract_path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in contract file: {e}")


@pytest.fixture
def published_events(contract_data: dict) -> list[dict]:
    """Extract published_events from contract data."""
    return contract_data.get("published_events", [])


@pytest.fixture
def event_types_map(published_events: list[dict]) -> dict[str, dict]:
    """Build a map of event_type -> event definition for quick lookup."""
    return {event["event_type"]: event for event in published_events}


# =============================================================================
# Constants
# =============================================================================

# The 7 decision path event types as defined in OMN-888
DECISION_PATH_EVENT_TYPES = [
    "NodeRegistrationInitiated",
    "NodeRegistrationAccepted",
    "NodeRegistrationRejected",
    "NodeRegistrationAckTimedOut",
    "NodeRegistrationAckReceived",
    "NodeBecameActive",
    "NodeLivenessExpired",
]

# The result event (not a decision path, but a workflow outcome)
RESULT_EVENT_TYPE = "NodeRegistrationResultEvent"

# All published event types (7 decision + 1 result = 8 total)
ALL_PUBLISHED_EVENT_TYPES = DECISION_PATH_EVENT_TYPES + [RESULT_EVENT_TYPE]

# Topic pattern regex: {env}.{namespace}.onex.evt.<slug>.v1
# The placeholders {env} and {namespace} are variable, followed by literal pattern
TOPIC_PATTERN_REGEX = re.compile(r"^\{env\}\.\{namespace\}\.onex\.evt\.[a-z0-9-]+\.v1$")


# =============================================================================
# TestDecisionPathEvents
# =============================================================================


class TestDecisionPathEvents:
    """Tests for all 7 decision path event types defined in OMN-888.

    These tests verify that the orchestrator contract correctly declares
    all decision path events with proper topic patterns.
    """

    def test_contract_publishes_all_7_event_types(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that all 7 decision path event types are in published_events.

        The orchestrator must publish events for each possible decision path:
        - NodeRegistrationInitiated: Workflow start
        - NodeRegistrationAccepted: Registration success
        - NodeRegistrationRejected: Registration rejection
        - NodeRegistrationAckTimedOut: Acknowledgment timeout
        - NodeRegistrationAckReceived: Acknowledgment received
        - NodeBecameActive: Node activation
        - NodeLivenessExpired: Liveness failure

        This test ensures no decision path event is missing from the contract.
        """
        missing_events = []
        for event_type in DECISION_PATH_EVENT_TYPES:
            if event_type not in event_types_map:
                missing_events.append(event_type)

        assert not missing_events, (
            f"Missing decision path event types in published_events: {missing_events}\n"
            f"Expected all 7: {DECISION_PATH_EVENT_TYPES}\n"
            f"Found: {list(event_types_map.keys())}"
        )

    def test_node_registration_initiated_event_exists(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that NodeRegistrationInitiated event is properly defined.

        NodeRegistrationInitiated is emitted when a registration workflow starts.
        This event is triggered when the orchestrator receives an introspection
        event for a node that needs registration.

        Topic pattern: {env}.{namespace}.onex.evt.node-registration-initiated.v1
        """
        event_type = "NodeRegistrationInitiated"
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events"
        )

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-registration-initiated.v1"
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"
        )

    def test_node_registration_accepted_event_exists(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that NodeRegistrationAccepted event is properly defined.

        NodeRegistrationAccepted is emitted when a node's registration is
        successfully accepted. This occurs after both Consul and PostgreSQL
        registrations complete successfully.

        Topic pattern: {env}.{namespace}.onex.evt.node-registration-accepted.v1
        """
        event_type = "NodeRegistrationAccepted"
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events"
        )

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-registration-accepted.v1"
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"
        )

    def test_node_registration_rejected_event_exists(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that NodeRegistrationRejected event is properly defined.

        NodeRegistrationRejected is emitted when a node's registration is
        rejected. This can occur due to validation failures, policy violations,
        or when the node is not eligible for registration.

        Topic pattern: {env}.{namespace}.onex.evt.node-registration-rejected.v1
        """
        event_type = "NodeRegistrationRejected"
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events"
        )

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-registration-rejected.v1"
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"
        )

    def test_node_registration_ack_timed_out_event_exists(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that NodeRegistrationAckTimedOut event is properly defined.

        NodeRegistrationAckTimedOut is emitted when a node fails to acknowledge
        its registration within the configured timeout period. This indicates
        the node may be unresponsive or the acknowledgment message was lost.

        Topic pattern: {env}.{namespace}.onex.evt.node-registration-ack-timed-out.v1
        """
        event_type = "NodeRegistrationAckTimedOut"
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events"
        )

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-registration-ack-timed-out.v1"
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"
        )

    def test_node_registration_ack_received_event_exists(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that NodeRegistrationAckReceived event is properly defined.

        NodeRegistrationAckReceived is emitted when the orchestrator receives
        a successful acknowledgment from a registered node. This confirms the
        node has received and processed its registration information.

        Topic pattern: {env}.{namespace}.onex.evt.node-registration-ack-received.v1
        """
        event_type = "NodeRegistrationAckReceived"
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events"
        )

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-registration-ack-received.v1"
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"
        )

    def test_node_became_active_event_exists(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that NodeBecameActive event is properly defined.

        NodeBecameActive is emitted when a node transitions to the active state.
        This typically occurs after successful registration and acknowledgment,
        indicating the node is ready to participate in the system.

        Topic pattern: {env}.{namespace}.onex.evt.node-became-active.v1
        """
        event_type = "NodeBecameActive"
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events"
        )

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-became-active.v1"
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"
        )

    def test_node_liveness_expired_event_exists(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that NodeLivenessExpired event is properly defined.

        NodeLivenessExpired is emitted when a node's liveness check fails.
        This occurs when the node fails to send heartbeats or health checks
        within the configured timeout period, indicating potential failure.

        Topic pattern: {env}.{namespace}.onex.evt.node-liveness-expired.v1
        """
        event_type = "NodeLivenessExpired"
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events"
        )

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-liveness-expired.v1"
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"
        )

    def test_all_decision_events_follow_topic_convention(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that all decision events follow the ONEX topic naming convention.

        All published events must follow the pattern:
            {env}.{namespace}.onex.evt.<event-slug>.v1

        Where:
        - {env} and {namespace} are template placeholders
        - onex.evt is the literal event namespace
        - <event-slug> is a kebab-case event identifier
        - v1 is the version suffix

        This ensures consistent topic naming across the ONEX platform.
        """
        non_conforming_events = []

        for event_type in DECISION_PATH_EVENT_TYPES:
            if event_type not in event_types_map:
                continue  # Missing events are caught by other tests

            event = event_types_map[event_type]
            topic = event.get("topic", "")

            if not TOPIC_PATTERN_REGEX.match(topic):
                non_conforming_events.append(
                    f"{event_type}: '{topic}' does not match pattern "
                    f"'{{env}}.{{namespace}}.onex.evt.<slug>.v1'"
                )

        assert not non_conforming_events, (
            "Events with non-conforming topic patterns:\n"
            + "\n".join(f"  - {e}" for e in non_conforming_events)
        )

    def test_decision_event_count_is_exactly_8(
        self, published_events: list[dict]
    ) -> None:
        """Test that published_events has exactly 8 entries (7 decision + 1 result).

        The orchestrator must publish exactly 8 event types:
        - 7 decision path events (covering all workflow decision points)
        - 1 result event (NodeRegistrationResultEvent)

        This test ensures no events are missing and no unexpected events exist.
        Having more or fewer events indicates a contract definition error.
        """
        actual_count = len(published_events)
        expected_count = 8

        assert actual_count == expected_count, (
            f"published_events must have exactly {expected_count} entries "
            f"(7 decision events + 1 result event), found {actual_count}.\n"
            f"Expected event types: {ALL_PUBLISHED_EVENT_TYPES}\n"
            f"Found event types: {[e['event_type'] for e in published_events]}"
        )

        # Also verify all expected event types are present
        actual_event_types = {e["event_type"] for e in published_events}
        expected_event_types = set(ALL_PUBLISHED_EVENT_TYPES)

        missing = expected_event_types - actual_event_types
        extra = actual_event_types - expected_event_types

        assert not missing, f"Missing event types: {missing}"
        assert not extra, f"Unexpected event types: {extra}"


# =============================================================================
# TestResultEvent
# =============================================================================


class TestResultEvent:
    """Tests for the NodeRegistrationResultEvent (workflow outcome event).

    The result event is distinct from decision path events - it represents
    the final outcome of the registration workflow, not a decision point.
    """

    def test_result_event_exists(self, event_types_map: dict[str, dict]) -> None:
        """Test that NodeRegistrationResultEvent is properly defined.

        NodeRegistrationResultEvent contains the complete outcome of the
        registration workflow, including success/failure status, applied
        registrations, and any error information.

        Topic pattern: {env}.{namespace}.onex.evt.node-registration-result.v1
        """
        event_type = RESULT_EVENT_TYPE
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events"
        )

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-registration-result.v1"
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"
        )

    def test_result_event_follows_topic_convention(
        self, event_types_map: dict[str, dict]
    ) -> None:
        """Test that result event follows the ONEX topic naming convention."""
        if RESULT_EVENT_TYPE not in event_types_map:
            pytest.skip("Result event not defined")

        event = event_types_map[RESULT_EVENT_TYPE]
        topic = event.get("topic", "")

        assert TOPIC_PATTERN_REGEX.match(topic), (
            f"Result event topic '{topic}' does not match pattern "
            f"'{{env}}.{{namespace}}.onex.evt.<slug>.v1'"
        )


# =============================================================================
# TestEventStructure
# =============================================================================


class TestEventStructure:
    """Tests for proper event structure in the contract."""

    def test_all_events_have_required_fields(
        self, published_events: list[dict]
    ) -> None:
        """Test that all published events have required fields.

        Each event definition must include:
        - topic: The Kafka topic pattern for publishing
        - event_type: The event type name (matches model class name)
        """
        events_missing_fields = []

        for event in published_events:
            missing_fields = []
            if "topic" not in event:
                missing_fields.append("topic")
            if "event_type" not in event:
                missing_fields.append("event_type")

            if missing_fields:
                event_id = event.get("event_type", event.get("topic", "unknown"))
                events_missing_fields.append(f"{event_id}: missing {missing_fields}")

        assert not events_missing_fields, (
            "Events with missing required fields:\n"
            + "\n".join(f"  - {e}" for e in events_missing_fields)
        )

    def test_event_types_are_unique(self, published_events: list[dict]) -> None:
        """Test that all event types are unique (no duplicates)."""
        event_types = [e["event_type"] for e in published_events if "event_type" in e]
        duplicates = [et for et in event_types if event_types.count(et) > 1]

        assert not duplicates, f"Duplicate event types found: {list(set(duplicates))}"

    def test_topics_are_unique(self, published_events: list[dict]) -> None:
        """Test that all topics are unique (no duplicates)."""
        topics = [e["topic"] for e in published_events if "topic" in e]
        duplicates = [t for t in topics if topics.count(t) > 1]

        assert not duplicates, f"Duplicate topics found: {list(set(duplicates))}"
