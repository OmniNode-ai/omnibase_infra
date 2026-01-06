# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Idempotent Replay Tests for OMN-955.

This module verifies that idempotent replay produces consistent results.
Idempotency ensures that replaying the same events multiple times
yields identical final state without side effects.

Idempotency Guarantees:
    1. Same event processed multiple times = same final state
    2. Duplicate events emit no additional intents
    3. Replay count doesn't affect final state
    4. Event ID (correlation_id) is used for deduplication

Test Coverage:
    - Single event idempotency
    - Multi-replay idempotency
    - Sequence replay idempotency
    - Crash recovery scenarios
    - Intent emission on replays

Architecture:
    The RegistrationReducer implements idempotency via last_processed_event_id:

    1. Each event has a unique event_id (correlation_id or derived)
    2. After processing, state.last_processed_event_id is updated
    3. On replay, state.is_duplicate_event(event_id) returns True
    4. Duplicate events return current state unchanged with no intents

    This enables safe replay after crashes or network issues:
    - If crash occurs after state update but before Kafka ack
    - Event is redelivered
    - Reducer detects duplicate and skips
    - No duplicate intents are emitted

Related:
    - test_reducer_replay_determinism.py: Determinism tests
    - ModelRegistrationState.is_duplicate_event(): Idempotency check
    - OMN-955: Event Replay Verification ticket
"""

from __future__ import annotations

from datetime import UTC, datetime
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

if TYPE_CHECKING:
    from tests.replay.conftest import EventFactory


# =============================================================================
# Module-Level Markers
# =============================================================================
# These markers enable selective test execution:
#   pytest -m "replay" - run only replay tests
#   pytest -m "not replay" - skip replay tests

pytestmark = [
    pytest.mark.replay,
]

# =============================================================================
# Constants
# =============================================================================

EXPECTED_REGISTRATION_INTENTS = 2


# =============================================================================
# Single Event Idempotency Tests
# =============================================================================


@pytest.mark.unit
class TestSingleEventIdempotency:
    """Tests for single event idempotency."""

    def test_duplicate_event_returns_same_state(
        self,
        reducer: RegistrationReducer,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that duplicate event returns unchanged state.

        The core idempotency guarantee: processing the same event twice
        should return the same state the second time.
        """
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={"health": "http://localhost:8080/health"},
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        # First processing
        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event)

        # Second processing (duplicate) - use the result state
        output2 = reducer.reduce(output1.result, event)

        # State should be unchanged
        assert output2.result == output1.result
        assert output2.result.status == output1.result.status
        assert output2.result.node_id == output1.result.node_id
        assert (
            output2.result.last_processed_event_id
            == output1.result.last_processed_event_id
        )

    def test_duplicate_event_emits_no_intents(
        self,
        reducer: RegistrationReducer,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that duplicate event emits no intents.

        Critical for avoiding duplicate side effects in the Effect layer.
        """
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={},
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event)

        # Verify first processing emits intents
        assert len(output1.intents) == EXPECTED_REGISTRATION_INTENTS

        # Second processing (duplicate)
        output2 = reducer.reduce(output1.result, event)

        # No intents on duplicate
        assert len(output2.intents) == 0

    def test_duplicate_event_items_processed_zero(
        self,
        reducer: RegistrationReducer,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that duplicate event has items_processed=0.

        The items_processed field should reflect that no work was done.
        """
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={},
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event)

        # First processing should have items_processed=1
        assert output1.items_processed == 1

        # Second processing (duplicate)
        output2 = reducer.reduce(output1.result, event)

        # Duplicate should have items_processed=0
        assert output2.items_processed == 0


# =============================================================================
# Multi-Replay Idempotency Tests
# =============================================================================


@pytest.mark.unit
class TestMultiReplayIdempotency:
    """Tests for idempotency across multiple replays."""

    def test_multiple_replays_same_state(
        self,
        reducer: RegistrationReducer,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that multiple replays produce identical state.

        Replaying an event many times should always return the same state.
        """
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="compute",
            node_version="1.0.0",
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={"health": "http://localhost:8080/health"},
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        # Initial processing
        state = ModelRegistrationState()
        output = reducer.reduce(state, event)
        initial_result = output.result

        # Replay 10 times
        current_state = initial_result
        for _ in range(10):
            output = reducer.reduce(current_state, event)
            current_state = output.result

        # Final state should equal initial result
        assert current_state == initial_result

    def test_replay_count_does_not_affect_state(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that the number of replays doesn't affect final state.

        Whether replayed 1 time or 100 times, the state should be the same.
        """
        event = event_factory.create_event()

        # Single replay
        state1 = ModelRegistrationState()
        output1 = reducer.reduce(state1, event)
        single_replay_state = output1.result

        # 50 replays
        state2 = ModelRegistrationState()
        output2 = reducer.reduce(state2, event)
        current_state = output2.result

        for _ in range(49):
            output2 = reducer.reduce(current_state, event)
            current_state = output2.result

        # States should be equal
        assert current_state == single_replay_state

    def test_multiple_replays_no_cumulative_intents(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that multiple replays don't accumulate intents.

        Total intents should be 2 (from first processing), not 2*N.
        """
        event = event_factory.create_event()

        state = ModelRegistrationState()
        total_intents = 0

        # Initial processing
        output = reducer.reduce(state, event)
        total_intents += len(output.intents)
        state = output.result

        # 10 more replays
        for _ in range(10):
            output = reducer.reduce(state, event)
            total_intents += len(output.intents)
            state = output.result

        # Only first processing should emit intents
        assert total_intents == EXPECTED_REGISTRATION_INTENTS


# =============================================================================
# Sequence Replay Idempotency Tests
# =============================================================================


@pytest.mark.unit
class TestSequenceReplayIdempotency:
    """Tests for sequence replay idempotency."""

    def test_sequence_replay_produces_same_final_state(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that replaying a sequence produces same final states.

        Each event in the sequence should maintain idempotency.
        """
        events = event_factory.create_event_sequence(count=5)

        # First pass: process all events
        first_pass_states: list[ModelRegistrationState] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            first_pass_states.append(output.result)

        # Second pass: replay all events with their result states
        second_pass_states: list[ModelRegistrationState] = []
        for event, first_state in zip(events, first_pass_states, strict=True):
            output = reducer.reduce(first_state, event)
            second_pass_states.append(output.result)

        # States should be unchanged (idempotent)
        for i, (first, second) in enumerate(
            zip(first_pass_states, second_pass_states, strict=True)
        ):
            assert first == second, f"Event {i} replay changed state"

    def test_sequence_replay_no_duplicate_intents(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that sequence replay doesn't emit duplicate intents.

        Total intents should be 2*N (N events), not more.
        """
        events = event_factory.create_event_sequence(count=5)

        # First pass
        first_pass_intent_count = 0
        first_pass_states: list[ModelRegistrationState] = []

        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            first_pass_intent_count += len(output.intents)
            first_pass_states.append(output.result)

        # Second pass (replay)
        second_pass_intent_count = 0
        for event, first_state in zip(events, first_pass_states, strict=True):
            output = reducer.reduce(first_state, event)
            second_pass_intent_count += len(output.intents)

        # First pass should emit all intents
        assert first_pass_intent_count == 5 * EXPECTED_REGISTRATION_INTENTS

        # Second pass should emit no intents (all duplicates)
        assert second_pass_intent_count == 0

    def test_interleaved_replay_idempotent(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test idempotency with interleaved event and replay.

        Process: E1, E1, E2, E2, E3, E3 should be same as E1, E2, E3.
        """
        events = event_factory.create_event_sequence(count=3)

        # Interleaved: each event processed twice in a row
        interleaved_states: list[ModelRegistrationState] = []
        interleaved_intents = 0

        for event in events:
            state = ModelRegistrationState()

            # First time
            output = reducer.reduce(state, event)
            interleaved_intents += len(output.intents)

            # Immediate replay
            output = reducer.reduce(output.result, event)
            interleaved_intents += len(output.intents)

            interleaved_states.append(output.result)

        # Sequential: each event once
        sequential_states: list[ModelRegistrationState] = []
        sequential_intents = 0

        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            sequential_intents += len(output.intents)
            sequential_states.append(output.result)

        # Final states should match
        for i, (inter, seq) in enumerate(
            zip(interleaved_states, sequential_states, strict=True)
        ):
            assert inter == seq, f"Event {i} interleaved state differs"

        # Intent counts should match (replays don't add intents)
        assert interleaved_intents == sequential_intents


# =============================================================================
# Crash Recovery Scenario Tests
# =============================================================================


@pytest.mark.unit
class TestCrashRecoveryIdempotency:
    """Tests for idempotency in crash recovery scenarios."""

    def test_crash_after_state_update_replay_safe(
        self,
        reducer: RegistrationReducer,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test replay safety after crash following state update.

        Scenario:
        1. Event processed, state updated
        2. System crashes before Kafka ack
        3. Event redelivered
        4. Reducer should detect duplicate and skip
        """
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={"health": "http://localhost:8080/health"},
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        # Step 1: Event processed, state updated
        initial_state = ModelRegistrationState()
        output = reducer.reduce(initial_state, event)
        state_after_processing = output.result

        # Verify state was updated
        assert state_after_processing.status == "pending"
        assert state_after_processing.last_processed_event_id == fixed_correlation_id

        # Step 2 & 3: Simulate crash and redelivery
        # The state is persisted, but Kafka didn't ack
        # Event is redelivered, we process with saved state

        # Step 4: Reducer should detect duplicate
        redelivery_output = reducer.reduce(state_after_processing, event)

        # Should be idempotent
        assert redelivery_output.result == state_after_processing
        assert len(redelivery_output.intents) == 0
        assert redelivery_output.items_processed == 0

    def test_multiple_redeliveries_after_crash(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test handling of multiple redeliveries after crash.

        Network issues might cause multiple redeliveries of the same event.
        """
        event = event_factory.create_event()

        # Initial processing
        state = ModelRegistrationState()
        output = reducer.reduce(state, event)
        saved_state = output.result

        # Simulate 5 redeliveries
        redelivery_results: list[int] = []  # Track intent counts
        current_state = saved_state

        for _ in range(5):
            output = reducer.reduce(current_state, event)
            redelivery_results.append(len(output.intents))
            current_state = output.result

        # All redeliveries should emit 0 intents
        assert all(count == 0 for count in redelivery_results)

        # Final state should equal saved state
        assert current_state == saved_state

    def test_partial_batch_recovery(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test recovery from partial batch processing.

        Scenario: Batch of events, crash after some processed.
        Entire batch is replayed; already-processed events should be skipped.
        """
        events = event_factory.create_event_sequence(count=5)

        # Process first 3 events (simulating partial batch before crash)
        processed_states: list[ModelRegistrationState] = []
        for event in events[:3]:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            processed_states.append(output.result)

        # Now replay entire batch (simulating recovery)
        recovery_intent_count = 0
        recovery_states: list[ModelRegistrationState] = []

        for i, event in enumerate(events):
            if i < 3:
                # Already processed - use saved state
                state = processed_states[i]
            else:
                # Not yet processed - use fresh state
                state = ModelRegistrationState()

            output = reducer.reduce(state, event)
            recovery_intent_count += len(output.intents)
            recovery_states.append(output.result)

        # First 3 should have been skipped (0 intents each)
        # Last 2 should have been processed (2 intents each)
        expected_intents = 2 * EXPECTED_REGISTRATION_INTENTS
        assert recovery_intent_count == expected_intents


# =============================================================================
# Event ID Deduplication Tests
# =============================================================================


@pytest.mark.unit
class TestEventIdDeduplication:
    """Tests for event ID-based deduplication."""

    def test_correlation_id_used_for_deduplication(
        self,
        reducer: RegistrationReducer,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that correlation_id is used for deduplication.

        Events with the same correlation_id should be deduplicated.
        """
        correlation_id = UUID("11111111-1111-1111-1111-111111111111")

        # Two events with same correlation_id but different node_ids
        event1 = ModelNodeIntrospectionEvent(
            node_id=UUID("22222222-2222-2222-2222-222222222222"),
            node_type="effect",
            node_version="1.0.0",
            correlation_id=correlation_id,
            timestamp=fixed_timestamp,
            endpoints={},
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        event2 = ModelNodeIntrospectionEvent(
            node_id=UUID("33333333-3333-3333-3333-333333333333"),
            node_type="effect",
            node_version="1.0.0",
            correlation_id=correlation_id,  # Same correlation_id
            timestamp=fixed_timestamp,
            endpoints={},
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        # Process event1
        state = ModelRegistrationState()
        output1 = reducer.reduce(state, event1)

        # Process event2 with result state (same correlation_id)
        output2 = reducer.reduce(output1.result, event2)

        # event2 should be treated as duplicate
        assert len(output2.intents) == 0

    def test_different_correlation_ids_not_deduplicated(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that events with different correlation_ids are not deduplicated.

        Each unique correlation_id represents a distinct event.
        """
        events = event_factory.create_event_sequence(count=3)

        # All events should have unique correlation_ids
        correlation_ids = {event.correlation_id for event in events}
        assert len(correlation_ids) == 3

        # All should be processed (each with fresh state)
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            assert len(output.intents) == EXPECTED_REGISTRATION_INTENTS

    def test_last_processed_event_id_updated_correctly(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that last_processed_event_id is updated after processing.

        The state should track which event was last processed.
        """
        event = event_factory.create_event()

        initial_state = ModelRegistrationState()
        assert initial_state.last_processed_event_id is None

        output = reducer.reduce(initial_state, event)

        # last_processed_event_id should be set
        assert output.result.last_processed_event_id is not None
        assert output.result.last_processed_event_id == event.correlation_id


# =============================================================================
# Intent Emission Control Tests
# =============================================================================


@pytest.mark.unit
class TestIntentEmissionControl:
    """Tests for controlling intent emission in replay scenarios."""

    def test_first_processing_emits_expected_intents(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that first processing emits expected intents.

        Baseline: first processing should emit Consul and PostgreSQL intents.
        """
        event = event_factory.create_event()

        state = ModelRegistrationState()
        output = reducer.reduce(state, event)

        assert len(output.intents) == EXPECTED_REGISTRATION_INTENTS

        # Verify extension intent types
        extension_types = {
            intent.payload.extension_type
            for intent in output.intents
            if intent.intent_type == "extension"
        }
        assert "infra.consul_register" in extension_types
        assert "infra.postgres_upsert" in extension_types

    def test_replay_never_emits_intents(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that replays never emit intents.

        After initial processing, all replays should emit 0 intents.
        """
        event = event_factory.create_event()

        # Initial processing
        state = ModelRegistrationState()
        output = reducer.reduce(state, event)
        current_state = output.result

        # Many replays - none should emit intents
        for i in range(20):
            output = reducer.reduce(current_state, event)
            assert len(output.intents) == 0, f"Replay {i + 1} emitted intents"
            current_state = output.result

    def test_intent_count_matches_unique_events_only(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that total intents match unique events processed.

        Total intents = unique_events * 2 (Consul + PostgreSQL per event).
        """
        unique_events = event_factory.create_event_sequence(count=5)

        total_intents = 0

        for event in unique_events:
            # First time - fresh state
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            total_intents += len(output.intents)

            # Replay - same state
            output = reducer.reduce(output.result, event)
            total_intents += len(output.intents)

        # Should only count unique events
        expected = 5 * EXPECTED_REGISTRATION_INTENTS
        assert total_intents == expected
