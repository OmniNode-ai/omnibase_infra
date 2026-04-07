# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the build loop reducer FSM.

Tests the delta function: state transitions, circuit breaker, deduplication.

Related:
    - OMN-7313: node_loop_state_reducer
    - OMN-7323: Canary integration test
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_build_loop_intent_type import EnumBuildLoopIntentType
from omnibase_infra.enums.enum_build_loop_phase import EnumBuildLoopPhase
from omnibase_infra.nodes.node_loop_state_reducer.handlers.handler_loop_state import (
    HandlerLoopState,
)
from omnibase_infra.nodes.node_loop_state_reducer.models.model_build_loop_event import (
    ModelBuildLoopEvent,
)
from omnibase_infra.nodes.node_loop_state_reducer.models.model_build_loop_state import (
    ModelBuildLoopState,
)


@pytest.fixture
def reducer() -> HandlerLoopState:
    return HandlerLoopState()


@pytest.fixture
def idle_state() -> ModelBuildLoopState:
    return ModelBuildLoopState(
        correlation_id=uuid4(),
        phase=EnumBuildLoopPhase.IDLE,
    )


def _event(
    state: ModelBuildLoopState,
    success: bool = True,
    error_message: str | None = None,
    **kwargs: object,
) -> ModelBuildLoopEvent:
    """Helper to create an event matching the current state."""
    return ModelBuildLoopEvent(
        correlation_id=state.correlation_id,
        source_phase=state.phase,
        success=success,
        timestamp=datetime.now(tz=UTC),
        error_message=error_message,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.unit
class TestReducerHappyPath:
    """Test the full happy path through all phases."""

    def test_idle_to_closing_out(
        self, reducer: HandlerLoopState, idle_state: ModelBuildLoopState
    ):
        new_state, intents = reducer.delta(idle_state, _event(idle_state))
        assert new_state.phase == EnumBuildLoopPhase.CLOSING_OUT
        assert new_state.cycle_number == 1
        assert len(intents) == 1
        assert intents[0].intent_type == EnumBuildLoopIntentType.START_CLOSEOUT

    def test_idle_to_verifying_with_skip_closeout(self, reducer: HandlerLoopState):
        state = ModelBuildLoopState(
            correlation_id=uuid4(),
            phase=EnumBuildLoopPhase.IDLE,
            skip_closeout=True,
        )
        new_state, intents = reducer.delta(state, _event(state))
        assert new_state.phase == EnumBuildLoopPhase.VERIFYING
        assert len(intents) == 1
        assert intents[0].intent_type == EnumBuildLoopIntentType.START_VERIFY

    def test_full_cycle(
        self, reducer: HandlerLoopState, idle_state: ModelBuildLoopState
    ):
        """Walk through IDLE -> CLOSING_OUT -> DEPLOYING -> VERIFYING -> FILLING -> CLASSIFYING -> BUILDING -> COMPLETE."""
        state = idle_state

        # IDLE -> CLOSING_OUT
        state, intents = reducer.delta(state, _event(state))
        assert state.phase == EnumBuildLoopPhase.CLOSING_OUT

        # CLOSING_OUT -> DEPLOYING
        state, intents = reducer.delta(state, _event(state))
        assert state.phase == EnumBuildLoopPhase.DEPLOYING

        # DEPLOYING -> VERIFYING
        state, intents = reducer.delta(state, _event(state))
        assert state.phase == EnumBuildLoopPhase.VERIFYING

        # VERIFYING -> FILLING
        state, intents = reducer.delta(state, _event(state))
        assert state.phase == EnumBuildLoopPhase.FILLING

        # FILLING -> CLASSIFYING
        state, intents = reducer.delta(state, _event(state, tickets_filled=5))
        assert state.phase == EnumBuildLoopPhase.CLASSIFYING
        assert state.tickets_filled == 5

        # CLASSIFYING -> BUILDING
        state, intents = reducer.delta(state, _event(state, tickets_classified=3))
        assert state.phase == EnumBuildLoopPhase.BUILDING
        assert state.tickets_classified == 3

        # BUILDING -> COMPLETE
        state, intents = reducer.delta(state, _event(state, tickets_dispatched=3))
        assert state.phase == EnumBuildLoopPhase.COMPLETE
        assert state.tickets_dispatched == 3
        assert len(intents) == 1
        assert intents[0].intent_type == EnumBuildLoopIntentType.CYCLE_COMPLETE


@pytest.mark.unit
class TestReducerFailure:
    """Test failure transitions and circuit breaker."""

    def test_single_failure(self, reducer: HandlerLoopState):
        state = ModelBuildLoopState(
            correlation_id=uuid4(),
            phase=EnumBuildLoopPhase.VERIFYING,
            cycle_number=1,
        )
        new_state, intents = reducer.delta(
            state, _event(state, success=False, error_message="Health check failed")
        )
        assert new_state.phase == EnumBuildLoopPhase.FAILED
        assert new_state.consecutive_failures == 1
        assert new_state.error_message == "Health check failed"
        assert intents == []

    def test_circuit_breaker_trips(self, reducer: HandlerLoopState):
        """After max_consecutive_failures, emit CIRCUIT_BREAK intent."""
        state = ModelBuildLoopState(
            correlation_id=uuid4(),
            phase=EnumBuildLoopPhase.VERIFYING,
            cycle_number=1,
            consecutive_failures=2,
            max_consecutive_failures=3,
        )
        new_state, intents = reducer.delta(
            state, _event(state, success=False, error_message="Third failure")
        )
        assert new_state.phase == EnumBuildLoopPhase.FAILED
        assert new_state.consecutive_failures == 3
        assert len(intents) == 1
        assert intents[0].intent_type == EnumBuildLoopIntentType.CIRCUIT_BREAK


@pytest.mark.unit
class TestReducerDeduplication:
    """Test duplicate/out-of-order event rejection."""

    def test_wrong_correlation_id(
        self, reducer: HandlerLoopState, idle_state: ModelBuildLoopState
    ):
        wrong_event = ModelBuildLoopEvent(
            correlation_id=uuid4(),  # different from state
            source_phase=EnumBuildLoopPhase.IDLE,
            success=True,
            timestamp=datetime.now(tz=UTC),
        )
        new_state, intents = reducer.delta(idle_state, wrong_event)
        assert new_state is idle_state  # unchanged
        assert intents == []

    def test_wrong_source_phase(self, reducer: HandlerLoopState):
        state = ModelBuildLoopState(
            correlation_id=uuid4(),
            phase=EnumBuildLoopPhase.VERIFYING,
            cycle_number=1,
        )
        wrong_event = ModelBuildLoopEvent(
            correlation_id=state.correlation_id,
            source_phase=EnumBuildLoopPhase.IDLE,  # wrong phase
            success=True,
            timestamp=datetime.now(tz=UTC),
        )
        new_state, intents = reducer.delta(state, wrong_event)
        assert new_state is state  # unchanged
        assert intents == []

    def test_terminal_state_rejects_events(self, reducer: HandlerLoopState):
        state = ModelBuildLoopState(
            correlation_id=uuid4(),
            phase=EnumBuildLoopPhase.COMPLETE,
            cycle_number=1,
        )
        new_state, intents = reducer.delta(state, _event(state))
        assert new_state is state
        assert intents == []
