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

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================
# Note: The following fixtures are provided by conftest.py with module-level
# scope for performance (parse once per module):
#   - contract_path, contract_data: Contract loading
#   - published_events: List of published events from contract
#   - event_types_map: Map of event_type -> event definition


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

# Event type to expected topic mapping for parametrized tests
# Each tuple: (event_type, expected_topic, description)
# Description is used for test documentation and error messages
DECISION_EVENT_TOPIC_MAPPING = [
    (
        "NodeRegistrationInitiated",
        "{env}.{namespace}.onex.evt.node-registration-initiated.v1",
        "Emitted when a registration workflow starts (introspection event received)",
    ),
    (
        "NodeRegistrationAccepted",
        "{env}.{namespace}.onex.evt.node-registration-accepted.v1",
        "Emitted when registration is successfully accepted (Consul + PostgreSQL success)",
    ),
    (
        "NodeRegistrationRejected",
        "{env}.{namespace}.onex.evt.node-registration-rejected.v1",
        "Emitted when registration is rejected (validation/policy failures)",
    ),
    (
        "NodeRegistrationAckTimedOut",
        "{env}.{namespace}.onex.evt.node-registration-ack-timed-out.v1",
        "Emitted when acknowledgment times out (node unresponsive)",
    ),
    (
        "NodeRegistrationAckReceived",
        "{env}.{namespace}.onex.evt.node-registration-ack-received.v1",
        "Emitted when acknowledgment is received (node confirmed registration)",
    ),
    (
        "NodeBecameActive",
        "{env}.{namespace}.onex.evt.node-became-active.v1",
        "Emitted when node transitions to active state (ready to participate)",
    ),
    (
        "NodeLivenessExpired",
        "{env}.{namespace}.onex.evt.node-liveness-expired.v1",
        "Emitted when liveness check fails (heartbeat/health timeout)",
    ),
]


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

    @pytest.mark.parametrize(
        ("event_type", "expected_topic", "description"),
        DECISION_EVENT_TOPIC_MAPPING,
        ids=[event[0] for event in DECISION_EVENT_TOPIC_MAPPING],
    )
    def test_decision_event_exists_with_correct_topic(
        self,
        event_types_map: dict[str, dict],
        event_type: str,
        expected_topic: str,
        description: str,
    ) -> None:
        """Test that each decision path event is properly defined with correct topic.

        This parametrized test validates each of the 7 decision path event types:
        - NodeRegistrationInitiated: Workflow start
        - NodeRegistrationAccepted: Registration success
        - NodeRegistrationRejected: Registration rejection
        - NodeRegistrationAckTimedOut: Acknowledgment timeout
        - NodeRegistrationAckReceived: Acknowledgment received
        - NodeBecameActive: Node activation
        - NodeLivenessExpired: Liveness failure

        Each event must:
        1. Be defined in published_events
        2. Have a topic matching the ONEX pattern: {env}.{namespace}.onex.evt.<slug>.v1

        Args:
            event_types_map: Mapping of event_type -> event definition from contract.
            event_type: The event type name to test.
            expected_topic: The expected topic pattern for this event.
            description: Human-readable description of when this event is emitted.
        """
        # Verify event exists in published_events
        assert event_type in event_types_map, (
            f"{event_type} must be defined in published_events.\n"
            f"Description: {description}"
        )

        # Verify topic matches expected pattern
        event = event_types_map[event_type]
        assert event["topic"] == expected_topic, (
            f"{event_type} topic should be '{expected_topic}', "
            f"got '{event['topic']}'.\n"
            f"Description: {description}"
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
        assert (
            event_type in event_types_map
        ), f"{event_type} must be defined in published_events"

        event = event_types_map[event_type]
        expected_topic = "{env}.{namespace}.onex.evt.node-registration-result.v1"
        assert (
            event["topic"] == expected_topic
        ), f"{event_type} topic should be '{expected_topic}', got '{event['topic']}'"

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


# =============================================================================
# TestConsumedEventHandlers
# =============================================================================


class TestConsumedEventHandlers:
    """Tests for contract consistency between consumed events and workflow handlers.

    This test class validates that every event declared in consumed_events has
    a corresponding receive step in the execution graph. This ensures contract
    completeness - if we declare we consume an event, there must be workflow
    logic that handles it.
    """

    @staticmethod
    def _extract_topic_slug(topic: str) -> str:
        """Extract the event slug from a topic pattern.

        Topic patterns follow: {env}.{namespace}.onex.evt.<slug>.v1
        or: {env}.{namespace}.onex.internal.<slug>.v1

        Args:
            topic: The full topic pattern string.

        Returns:
            The extracted slug (e.g., 'node-introspection' from the topic).
        """
        # Remove template placeholders and split by dots
        parts = topic.split(".")
        # The slug is typically the second-to-last part (before version)
        # Pattern: {env}.{namespace}.onex.(evt|internal).<slug>.v1
        if len(parts) >= 2:
            return parts[-2]  # e.g., 'node-introspection', 'runtime-tick'
        return topic

    @staticmethod
    def _pattern_matches_slug(pattern: str, slug: str) -> bool:
        """Check if an event pattern matches a topic slug.

        Event patterns use dot-separated segments with wildcards.
        Examples:
            - 'node.introspection.*' matches 'node-introspection'
            - 'runtime-tick.*' matches 'runtime-tick'

        The matching logic:
            1. Remove trailing wildcard from pattern
            2. Convert pattern dots to dashes for comparison
            3. Check if slug starts with the normalized pattern

        Args:
            pattern: Event pattern from step_config (e.g., 'node.introspection.*').
            slug: Topic slug to match (e.g., 'node-introspection').

        Returns:
            True if the pattern matches the slug.
        """
        # Remove trailing wildcard
        normalized_pattern = pattern.rstrip("*").rstrip(".")

        # Convert pattern dots to dashes for slug comparison
        normalized_pattern = normalized_pattern.replace(".", "-")

        # Check if slug starts with the pattern prefix
        return slug.startswith(normalized_pattern)

    def test_consumed_events_have_workflow_handlers(
        self, contract_data: dict, execution_graph_nodes: list[dict]
    ) -> None:
        """Ensure every consumed event has a corresponding receive step.

        This validates contract consistency - if we declare we consume an event,
        there must be a workflow step that handles it. Events without handlers
        indicate either:
            1. A missing receive step in the execution graph
            2. An event that should be removed from consumed_events

        The test extracts event patterns from all effect nodes that have
        event_pattern in their step_config, then validates each consumed
        event matches at least one pattern.
        """
        # Get consumed events
        consumed_events = contract_data.get("consumed_events", [])
        if not consumed_events:
            pytest.skip("No consumed_events defined in contract")

        # Extract topic slugs from consumed events
        consumed_slugs: dict[str, str] = {}  # slug -> event_type for error messages
        for event in consumed_events:
            topic = event.get("topic", "")
            event_type = event.get("event_type", "unknown")
            if topic:
                slug = self._extract_topic_slug(topic)
                consumed_slugs[slug] = event_type

        # Collect all event patterns from execution graph nodes
        handled_patterns: list[str] = []
        for node in execution_graph_nodes:
            step_config = node.get("step_config", {})
            event_pattern = step_config.get("event_pattern")

            if event_pattern:
                # event_pattern can be a list or a string
                if isinstance(event_pattern, list):
                    handled_patterns.extend(event_pattern)
                else:
                    handled_patterns.append(event_pattern)

        if not handled_patterns:
            pytest.fail(
                "No event_pattern found in any execution_graph node step_config. "
                "At least one receive step should declare event patterns."
            )

        # Validate each consumed event matches at least one handler pattern
        unhandled: list[str] = []
        for slug, event_type in consumed_slugs.items():
            matches = any(
                self._pattern_matches_slug(pattern, slug)
                for pattern in handled_patterns
            )
            if not matches:
                unhandled.append(f"{event_type} (slug: {slug})")

        assert not unhandled, (
            f"Consumed events without workflow handlers: {unhandled}\n"
            f"Every consumed event must have a corresponding receive step "
            f"with matching event_pattern in execution_graph.\n"
            f"Available patterns: {handled_patterns}\n"
            f"Either add a receive step with matching pattern, "
            f"or remove the event from consumed_events."
        )

    def test_consumed_events_section_exists(self, contract_data: dict) -> None:
        """Test that consumed_events section is defined in the contract.

        An orchestrator must declare which events it consumes to enable
        proper event routing and subscription management.
        """
        assert "consumed_events" in contract_data, (
            "Contract must define 'consumed_events' section. "
            "This declares which events the orchestrator subscribes to."
        )

    def test_consumed_events_have_required_fields(self, contract_data: dict) -> None:
        """Test that all consumed events have required fields.

        Each consumed event must include:
            - topic: The Kafka topic pattern to subscribe to
            - event_type: The event type name for deserialization
        """
        consumed_events = contract_data.get("consumed_events", [])
        events_missing_fields: list[str] = []

        for event in consumed_events:
            missing_fields: list[str] = []
            if "topic" not in event:
                missing_fields.append("topic")
            if "event_type" not in event:
                missing_fields.append("event_type")

            if missing_fields:
                event_id = event.get("event_type", event.get("topic", "unknown"))
                events_missing_fields.append(f"{event_id}: missing {missing_fields}")

        assert not events_missing_fields, (
            "Consumed events with missing required fields:\n"
            + "\n".join(f"  - {e}" for e in events_missing_fields)
        )
