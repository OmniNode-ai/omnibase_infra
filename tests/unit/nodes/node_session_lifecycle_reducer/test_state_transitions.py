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
    # Invalid transition enforcement (all 12 invalid pairs)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("source_status", "transition_method", "expected_match"),
        [
            # with_run_created: only valid from IDLE
            (
                EnumSessionLifecycleState.RUN_CREATED,
                "with_run_created",
                "requires IDLE",
            ),
            (EnumSessionLifecycleState.RUN_ACTIVE, "with_run_created", "requires IDLE"),
            (EnumSessionLifecycleState.RUN_ENDED, "with_run_created", "requires IDLE"),
            # with_run_activated: only valid from RUN_CREATED
            (
                EnumSessionLifecycleState.IDLE,
                "with_run_activated",
                "requires RUN_CREATED",
            ),
            (
                EnumSessionLifecycleState.RUN_ACTIVE,
                "with_run_activated",
                "requires RUN_CREATED",
            ),
            (
                EnumSessionLifecycleState.RUN_ENDED,
                "with_run_activated",
                "requires RUN_CREATED",
            ),
            # with_run_ended: only valid from RUN_ACTIVE
            (EnumSessionLifecycleState.IDLE, "with_run_ended", "requires RUN_ACTIVE"),
            (
                EnumSessionLifecycleState.RUN_CREATED,
                "with_run_ended",
                "requires RUN_ACTIVE",
            ),
            (
                EnumSessionLifecycleState.RUN_ENDED,
                "with_run_ended",
                "requires RUN_ACTIVE",
            ),
            # with_reset: only valid from RUN_ENDED
            (EnumSessionLifecycleState.IDLE, "with_reset", "requires RUN_ENDED"),
            (EnumSessionLifecycleState.RUN_CREATED, "with_reset", "requires RUN_ENDED"),
            (EnumSessionLifecycleState.RUN_ACTIVE, "with_reset", "requires RUN_ENDED"),
        ],
        ids=[
            "run_created-from-RUN_CREATED",
            "run_created-from-RUN_ACTIVE",
            "run_created-from-RUN_ENDED",
            "run_activated-from-IDLE",
            "run_activated-from-RUN_ACTIVE",
            "run_activated-from-RUN_ENDED",
            "run_ended-from-IDLE",
            "run_ended-from-RUN_CREATED",
            "run_ended-from-RUN_ENDED",
            "reset-from-IDLE",
            "reset-from-RUN_CREATED",
            "reset-from-RUN_ACTIVE",
        ],
    )
    def test_invalid_transition_raises(
        self,
        source_status: EnumSessionLifecycleState,
        transition_method: str,
        expected_match: str,
    ) -> None:
        """Every invalid (source_state, transition) pair raises ValueError."""
        # States that need a run_id to construct validly
        needs_run_id = {
            EnumSessionLifecycleState.RUN_CREATED,
            EnumSessionLifecycleState.RUN_ACTIVE,
            EnumSessionLifecycleState.RUN_ENDED,
        }
        kwargs: dict[str, str | EnumSessionLifecycleState] = {"status": source_status}
        if source_status in needs_run_id:
            kwargs["run_id"] = "run-1"
        state = ModelSessionLifecycleState(**kwargs)  # type: ignore[arg-type]

        method = getattr(state, transition_method)
        with pytest.raises(ValueError, match=expected_match):
            if transition_method == "with_run_created":
                method("run-2", uuid4())
            else:
                method(uuid4())
