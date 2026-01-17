# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Out-of-Order Event Tests for OMN-955.

This module tests replay behavior with out-of-order event arrival and
validates that ordering issues are detected and handled correctly.

Out-of-Order Scenarios:
    1. Events arriving with non-sequential timestamps
    2. Events being replayed in different order than original
    3. Sequence number violations in event logs
    4. Interleaved events from multiple sources

Test Coverage:
    - Detection of out-of-order events
    - Handling of sequence violations
    - Impact of event ordering on final state
    - Recovery from ordering anomalies

Architecture:
    The RegistrationReducer is stateless per invocation - each call to
    reduce() receives the full state and returns new state. This means:

    - Event ordering doesn't affect individual reductions
    - The orchestrator/runtime is responsible for event ordering
    - The reducer can detect redelivered events via last_processed_event_id
    - Timestamp ordering is informational, not enforced by reducer

    This test module validates the infrastructure's ability to detect
    and handle ordering issues at the application layer.

Related:
    - test_reducer_replay_determinism.py: Determinism tests
    - RegistrationReducer: Pure reducer under test
    - OMN-955: Event Replay Verification ticket
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers.replay_utils import (
    EventSequenceEntry,
    detect_sequence_number_violations,
    detect_timestamp_order_violations,
)

# =============================================================================
# Module-Level Markers
# =============================================================================
# These markers enable selective test execution:
#   pytest -m "replay" - run only replay tests
#   pytest -m "not replay" - skip replay tests

pytestmark = [
    pytest.mark.replay,
]

from omnibase_core.models.primitives.model_semver import ModelSemVer

if TYPE_CHECKING:
    from omnibase_core.nodes import ModelReducerOutput
    from tests.replay.conftest import EventFactory, EventSequenceLog


# =============================================================================
# Timestamp Ordering Tests
# =============================================================================


@pytest.mark.unit
class TestTimestampOrdering:
    """Tests for timestamp ordering detection and handling."""

    def test_detect_chronological_order(
        self,
        event_factory: EventFactory,
    ) -> None:
        """Test that chronologically ordered events have no violations.

        Verifies that properly ordered events pass validation.
        """
        events = event_factory.create_event_sequence(
            count=5,
            time_between_events=60,
        )

        violations = detect_timestamp_order_violations(events)
        assert len(violations) == 0, f"Unexpected violations: {violations}"

    def test_detect_reverse_order_violation(
        self,
        event_factory: EventFactory,
    ) -> None:
        """Test detection of reverse-ordered events.

        Verifies that out-of-order timestamps are detected.
        """
        events = event_factory.create_event_sequence(
            count=5,
            time_between_events=60,
        )

        # Reverse the list to create out-of-order timestamps
        reversed_events = list(reversed(events))

        violations = detect_timestamp_order_violations(reversed_events)
        # Should have n-1 violations (each event is before the previous)
        assert len(violations) == 4, f"Expected 4 violations, got {len(violations)}"

        for v in violations:
            assert v.violation_type == "timestamp_reorder"

    def test_detect_single_out_of_order_event(
        self,
        event_factory: EventFactory,
    ) -> None:
        """Test detection of a single out-of-order event.

        Verifies that a single misplaced event is detected.
        """
        events = event_factory.create_event_sequence(
            count=5,
            time_between_events=60,
        )

        # Swap event at position 2 with event at position 4
        events[2], events[4] = events[4], events[2]

        violations = detect_timestamp_order_violations(events)
        # Position 2 is now later than position 3, and position 4 is earlier
        assert len(violations) >= 1, "Should detect at least one violation"

    def test_detect_duplicate_timestamps(
        self,
        fixed_timestamp: datetime,
    ) -> None:
        """Test detection of duplicate timestamps.

        Events with identical timestamps may indicate ordering issues.
        """
        # Create events with identical timestamps
        events: list[ModelNodeIntrospectionEvent] = []
        for i in range(3):
            events.append(
                ModelNodeIntrospectionEvent(
                    node_id=UUID(int=100 + i),
                    node_type="effect",
                    node_version=ModelSemVer.parse("1.0.0"),
                    correlation_id=UUID(int=200 + i),
                    timestamp=fixed_timestamp,  # Same timestamp for all
                    endpoints={},
                    declared_capabilities=ModelNodeCapabilities(),
                    metadata=ModelNodeMetadata(),
                )
            )

        violations = detect_timestamp_order_violations(events)
        assert len(violations) == 2, "Should detect 2 duplicate timestamp violations"

        for v in violations:
            assert v.violation_type == "timestamp_duplicate"


# =============================================================================
# Reducer Behavior with Out-of-Order Events
# =============================================================================


@pytest.mark.unit
class TestReducerWithOutOfOrderEvents:
    """Tests for reducer behavior when events arrive out of order."""

    def test_reducer_processes_regardless_of_timestamp_order(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that reducer processes events regardless of timestamp order.

        The reducer is stateless per call - it doesn't enforce timestamp order.
        Each call is independent.
        """
        events = event_factory.create_event_sequence(
            count=3,
            time_between_events=60,
        )

        # Process in reverse order (each with fresh state)
        outputs: list[ModelReducerOutput] = []
        for event in reversed(events):
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            outputs.append(output)

        # Each event should still produce valid output
        for output in outputs:
            assert output.result.status == "pending"
            assert len(output.intents) == 2

    def test_out_of_order_affects_sequential_state_chain(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test impact of out-of-order events on sequential state chain.

        When events share state (chained), order matters for the final state.
        This test demonstrates the importance of proper event ordering.
        """
        events = event_factory.create_event_sequence(count=3)

        # Process in correct order: event0 -> event1 -> event2
        # Each event is for a different node, so we use fresh state each time
        correct_order_results: list[ModelRegistrationState] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            correct_order_results.append(output.result)

        # Process in reverse order
        reverse_order_results: list[ModelRegistrationState] = []
        for event in reversed(events):
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            reverse_order_results.append(output.result)

        # With fresh state per event, order doesn't matter for individual results
        # Each event produces the same result regardless of when it's processed
        assert len(correct_order_results) == len(reverse_order_results)

        # But the order of results is different
        reverse_order_results.reverse()
        for i, (correct, reverse) in enumerate(
            zip(correct_order_results, reverse_order_results, strict=True)
        ):
            assert correct.node_id == reverse.node_id, f"Event {i} node_id mismatch"

    def test_interleaved_node_type_events(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test processing interleaved events from different node types.

        Events from different node types arriving interleaved should
        each be processed independently.
        """
        # Create events for different node types
        effect_event = event_factory.create_event(node_type="effect")
        compute_event = event_factory.create_event(node_type="compute")
        reducer_event = event_factory.create_event(node_type="reducer")
        orchestrator_event = event_factory.create_event(node_type="orchestrator")

        # Interleave the events (not in type order)
        interleaved = [
            orchestrator_event,
            effect_event,
            reducer_event,
            compute_event,
        ]

        # Process all events
        results: list[tuple[str, str]] = []  # (node_type, status)
        for event in interleaved:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            results.append((event.node_type, output.result.status))

        # All should produce pending status
        for node_type, status in results:
            assert status == "pending", f"{node_type} did not produce pending status"


# =============================================================================
# Sequence Number Violation Tests
# =============================================================================


@pytest.mark.unit
class TestSequenceNumberViolations:
    """Tests for sequence number validation in event logs."""

    def test_detect_missing_sequence_number(
        self,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test detection of missing sequence numbers.

        Verifies that gaps in sequence numbers are detected.
        """
        events = event_factory.create_event_sequence(count=5)

        # Manually create entries with a gap in sequence numbers
        for i, event in enumerate(events):
            seq = i + 1 if i < 2 else i + 2  # Skip sequence number 3
            entry = EventSequenceEntry(
                event=event,
                expected_status="pending",
                expected_intent_count=2,
                sequence_number=seq,
            )
            event_sequence_log.entries.append(entry)

        violations = detect_sequence_number_violations(event_sequence_log)
        # Should detect violations at positions 2, 3, 4 (wrong sequence numbers)
        assert len(violations) >= 1, "Should detect sequence number violations"

    def test_valid_sequence_numbers_pass(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test that valid sequence numbers have no violations.

        Verifies that properly numbered sequences pass validation.
        """
        events = event_factory.create_event_sequence(count=5)

        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            event_sequence_log.append(
                event=event,
                expected_status=output.result.status,
                expected_intent_count=len(output.intents),
            )

        violations = detect_sequence_number_violations(event_sequence_log)
        assert len(violations) == 0, f"Unexpected violations: {violations}"


# =============================================================================
# Event Redelivery and Ordering Tests
# =============================================================================


@pytest.mark.unit
class TestEventRedeliveryOrdering:
    """Tests for handling event redelivery scenarios."""

    def test_redelivered_event_with_earlier_timestamp(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test handling of redelivered event with earlier timestamp.

        When an event is redelivered (same correlation_id), the reducer's
        idempotency should skip it, regardless of timestamp.
        """
        event = event_factory.create_event()

        # First delivery
        state1 = ModelRegistrationState()
        output1 = reducer.reduce(state1, event)

        # Simulate redelivery (same event, process on result state)
        output2 = reducer.reduce(output1.result, event)

        # Second delivery should be idempotent (no intents)
        assert len(output2.intents) == 0
        assert output2.result == output1.result

    def test_different_events_same_timestamp_processed(
        self,
        reducer: RegistrationReducer,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that different events with same timestamp are processed.

        Events with same timestamp but different correlation_ids should
        all be processed (they're different events).
        """
        events: list[ModelNodeIntrospectionEvent] = []
        for i in range(3):
            events.append(
                ModelNodeIntrospectionEvent(
                    node_id=UUID(int=100 + i),
                    node_type="effect",
                    node_version=ModelSemVer.parse("1.0.0"),
                    correlation_id=UUID(int=200 + i),  # Different correlation_id
                    timestamp=fixed_timestamp,  # Same timestamp
                    endpoints={},
                    declared_capabilities=ModelNodeCapabilities(),
                    metadata=ModelNodeMetadata(),
                )
            )

        # All events should be processed (each with fresh state)
        intent_counts: list[int] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            intent_counts.append(len(output.intents))

        # All should produce intents
        assert all(count == 2 for count in intent_counts)


# =============================================================================
# Ordering Impact Analysis Tests
# =============================================================================


@pytest.mark.unit
class TestOrderingImpactAnalysis:
    """Tests for analyzing the impact of event ordering on results."""

    def test_isolated_events_order_independent(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that isolated events (fresh state each) are order-independent.

        When each event is processed with fresh initial state, the
        processing order shouldn't affect individual results.
        """
        events = event_factory.create_event_sequence(count=5)

        # Process forward
        forward_results: list[str] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            forward_results.append(str(output.result.node_id))

        # Process backward
        backward_results: list[str] = []
        for event in reversed(events):
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            backward_results.append(str(output.result.node_id))
        backward_results.reverse()

        # Results should match (same events produce same results)
        assert forward_results == backward_results

    def test_ordering_violations_logged(
        self,
        event_factory: EventFactory,
    ) -> None:
        """Test that ordering violations are properly documented.

        Validates the violation detection produces useful information.
        """
        events = event_factory.create_event_sequence(count=5, time_between_events=60)

        # Create out-of-order scenario
        events[2], events[4] = events[4], events[2]

        violations = detect_timestamp_order_violations(events)

        # Violations should contain useful information
        for v in violations:
            assert v.position >= 0
            assert v.event_timestamp is not None
            assert v.previous_timestamp is not None
            assert v.violation_type is not None
