# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Tests for session lifecycle reducer state transitions.

Verifies the FSM: idle -> run_created -> run_active -> run_ended -> idle
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumSessionLifecycleState
from omnibase_infra.nodes.node_session_lifecycle_reducer.models import (
    ModelSessionLifecycleState,
)


class TestModelSessionLifecycleState:
    """Tests for ModelSessionLifecycleState FSM transitions."""

    def test_default_state_is_idle(self) -> None:
        """Default state is IDLE."""
        state = ModelSessionLifecycleState()
        assert state.status == EnumSessionLifecycleState.IDLE
        assert state.run_id is None

    def test_idle_to_run_created(self) -> None:
        """idle -> run_created via with_run_created."""
        state = ModelSessionLifecycleState()
        event_id = uuid4()
        new_state = state.with_run_created("run-1", event_id)

        assert new_state.status == EnumSessionLifecycleState.RUN_CREATED
        assert new_state.run_id == "run-1"
        assert new_state.last_processed_event_id == event_id

        # Original is unchanged (immutable)
        assert state.status == EnumSessionLifecycleState.IDLE

    def test_run_created_to_run_active(self) -> None:
        """run_created -> run_active via with_run_activated."""
        state = ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_CREATED,
            run_id="run-1",
        )
        event_id = uuid4()
        new_state = state.with_run_activated(event_id)

        assert new_state.status == EnumSessionLifecycleState.RUN_ACTIVE
        assert new_state.run_id == "run-1"  # Preserved
        assert new_state.last_processed_event_id == event_id

    def test_run_active_to_run_ended(self) -> None:
        """run_active -> run_ended via with_run_ended."""
        state = ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_ACTIVE,
            run_id="run-1",
        )
        event_id = uuid4()
        new_state = state.with_run_ended(event_id)

        assert new_state.status == EnumSessionLifecycleState.RUN_ENDED
        assert new_state.run_id == "run-1"  # Preserved for cleanup
        assert new_state.last_processed_event_id == event_id

    def test_run_ended_to_idle(self) -> None:
        """run_ended -> idle via with_reset."""
        state = ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_ENDED,
            run_id="run-1",
        )
        event_id = uuid4()
        new_state = state.with_reset(event_id)

        assert new_state.status == EnumSessionLifecycleState.IDLE
        assert new_state.run_id is None  # Cleared
        assert new_state.last_processed_event_id == event_id

    def test_full_lifecycle(self) -> None:
        """Test complete lifecycle: idle -> created -> active -> ended -> idle."""
        state = ModelSessionLifecycleState()

        # Create run
        state = state.with_run_created("run-lifecycle", uuid4())
        assert state.status == EnumSessionLifecycleState.RUN_CREATED

        # Activate run
        state = state.with_run_activated(uuid4())
        assert state.status == EnumSessionLifecycleState.RUN_ACTIVE

        # End run
        state = state.with_run_ended(uuid4())
        assert state.status == EnumSessionLifecycleState.RUN_ENDED

        # Reset to idle
        state = state.with_reset(uuid4())
        assert state.status == EnumSessionLifecycleState.IDLE
        assert state.run_id is None

    def test_idempotent_replay(self) -> None:
        """Replaying the same event_id is detected as duplicate."""
        state = ModelSessionLifecycleState()
        event_id = uuid4()
        state = state.with_run_created("run-1", event_id)

        assert state.is_duplicate_event(event_id)
        assert not state.is_duplicate_event(uuid4())

    def test_can_create_run_guard(self) -> None:
        """can_create_run is True only in IDLE state."""
        idle = ModelSessionLifecycleState()
        assert idle.can_create_run()

        created = idle.with_run_created("r", uuid4())
        assert not created.can_create_run()

        active = created.with_run_activated(uuid4())
        assert not active.can_create_run()

        ended = active.with_run_ended(uuid4())
        assert not ended.can_create_run()

    def test_can_activate_run_guard(self) -> None:
        """can_activate_run is True only in RUN_CREATED state."""
        idle = ModelSessionLifecycleState()
        assert not idle.can_activate_run()

        created = idle.with_run_created("r", uuid4())
        assert created.can_activate_run()

        active = created.with_run_activated(uuid4())
        assert not active.can_activate_run()

    def test_can_end_run_guard(self) -> None:
        """can_end_run is True only in RUN_ACTIVE state."""
        state = ModelSessionLifecycleState()
        assert not state.can_end_run()

        state = state.with_run_created("r", uuid4())
        assert not state.can_end_run()

        state = state.with_run_activated(uuid4())
        assert state.can_end_run()

        state = state.with_run_ended(uuid4())
        assert not state.can_end_run()

    def test_can_reset_guard(self) -> None:
        """can_reset is True only in RUN_ENDED state."""
        state = ModelSessionLifecycleState()
        assert not state.can_reset()

        state = state.with_run_created("r", uuid4())
        assert not state.can_reset()

        state = state.with_run_activated(uuid4())
        assert not state.can_reset()

        state = state.with_run_ended(uuid4())
        assert state.can_reset()

    def test_immutability(self) -> None:
        """State model is frozen — assignment raises error."""
        state = ModelSessionLifecycleState()
        with pytest.raises(Exception):
            state.status = EnumSessionLifecycleState.RUN_ACTIVE  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValueError):
            ModelSessionLifecycleState(unknown="oops")  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # Invalid transition enforcement
    # ------------------------------------------------------------------

    def test_with_run_created_from_non_idle_raises(self) -> None:
        """with_run_created from non-IDLE state raises ValueError."""
        state = ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_ACTIVE,
            run_id="run-1",
        )
        with pytest.raises(ValueError, match="requires IDLE"):
            state.with_run_created("run-2", uuid4())

    def test_with_run_activated_from_non_created_raises(self) -> None:
        """with_run_activated from non-RUN_CREATED state raises ValueError."""
        state = ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.IDLE,
        )
        with pytest.raises(ValueError, match="requires RUN_CREATED"):
            state.with_run_activated(uuid4())

    def test_with_run_ended_from_non_active_raises(self) -> None:
        """with_run_ended from non-RUN_ACTIVE state raises ValueError."""
        state = ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_CREATED,
            run_id="run-1",
        )
        with pytest.raises(ValueError, match="requires RUN_ACTIVE"):
            state.with_run_ended(uuid4())

    def test_with_reset_from_non_ended_raises(self) -> None:
        """with_reset from non-RUN_ENDED state raises ValueError."""
        state = ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_ACTIVE,
            run_id="run-1",
        )
        with pytest.raises(ValueError, match="requires RUN_ENDED"):
            state.with_reset(uuid4())
