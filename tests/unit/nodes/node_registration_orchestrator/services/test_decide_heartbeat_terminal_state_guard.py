# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Failing tests for decide_heartbeat terminal-state guard (OMN-4819).

These tests assert that decide_heartbeat returns no_op when the node's
current registration state is terminal (LIVENESS_EXPIRED, REJECTED).

Currently FAILING: decide_heartbeat does not guard against terminal states.
It proceeds to extend the liveness deadline even when the node is in a
terminal state, which triggers spurious re-registration side effects in
the handler layer.

Fix target: OMN-4822 — add terminal-state guard to decide_heartbeat.

Related Tickets:
    - OMN-4817: UUID Type Mismatch & Stale-Registration Race Fix (epic)
    - OMN-4819: This test file (write failing test)
    - OMN-4822: Fix decide_heartbeat to return no_op for terminal states
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import ModelNodeCapabilities
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_context import (
    ModelReducerContext,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixed test time for deterministic testing
# ---------------------------------------------------------------------------
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_projection(
    state: EnumRegistrationState,
    *,
    entity_id=None,
    liveness_deadline=None,
    last_heartbeat_at=None,
):
    """Build a ModelRegistrationProjection in the given state."""
    eid = entity_id or uuid4()
    return ModelRegistrationProjection(
        entity_id=eid,
        domain="registration",
        current_state=state,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),
        ack_deadline=None,
        liveness_deadline=liveness_deadline,
        last_heartbeat_at=last_heartbeat_at,
        ack_timeout_emitted_at=None,
        liveness_timeout_emitted_at=None,
        last_applied_event_id=uuid4(),
        last_applied_offset=0,
        registered_at=TEST_NOW - timedelta(hours=1),
        updated_at=TEST_NOW - timedelta(minutes=5),
    )


def make_ctx(now=None, correlation_id=None):
    """Build a ModelReducerContext for testing."""
    return ModelReducerContext(
        correlation_id=correlation_id or uuid4(),
        now=now or TEST_NOW,
        tick_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# Failing tests: decide_heartbeat must return no_op for terminal states
# ---------------------------------------------------------------------------


class TestDecideHeartbeatTerminalStateGuard:
    """decide_heartbeat must return no_op immediately for terminal states.

    These tests are RED until OMN-4822 adds the terminal-state guard.
    After OMN-4822, all tests in this class should go GREEN.
    """

    def test_liveness_expired_returns_no_op(self) -> None:
        """Heartbeat for LIVENESS_EXPIRED node must return no_op.

        When a node's liveness has expired and the next heartbeat arrives,
        decide_heartbeat must NOT extend the liveness deadline. Doing so
        would trigger spurious re-registration in handler_node_heartbeat.

        Currently FAILS: decide_heartbeat returns action='emit' (UPDATE intent)
        instead of action='no_op'.
        """
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.LIVENESS_EXPIRED,
            entity_id=node_id,
        )
        ctx = make_ctx()

        decision = service.decide_heartbeat(
            projection=projection,
            node_id=node_id,
            heartbeat_timestamp=TEST_NOW,
            ctx=ctx,
        )

        assert decision.action == "no_op", (
            f"Expected no_op for LIVENESS_EXPIRED state, got action={decision.action!r}. "
            "decide_heartbeat must guard against terminal states (OMN-4822)."
        )

    def test_rejected_returns_no_op(self) -> None:
        """Heartbeat for REJECTED node must return no_op.

        A REJECTED node should never receive a heartbeat in normal operation,
        but if one arrives, decide_heartbeat must not process it.

        Currently FAILS: decide_heartbeat returns action='emit' (UPDATE intent)
        instead of action='no_op'.
        """
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.REJECTED,
            entity_id=node_id,
        )
        ctx = make_ctx()

        decision = service.decide_heartbeat(
            projection=projection,
            node_id=node_id,
            heartbeat_timestamp=TEST_NOW,
            ctx=ctx,
        )

        assert decision.action == "no_op", (
            f"Expected no_op for REJECTED state, got action={decision.action!r}. "
            "decide_heartbeat must guard against terminal states (OMN-4822)."
        )

    def test_no_op_reason_mentions_terminal(self) -> None:
        """The no_op reason for a terminal-state heartbeat must reference terminal state.

        This ensures the reason is informative for debugging and log monitoring.

        Currently FAILS: no_op is not returned at all for terminal states.
        """
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.LIVENESS_EXPIRED,
            entity_id=node_id,
        )
        ctx = make_ctx()

        decision = service.decide_heartbeat(
            projection=projection,
            node_id=node_id,
            heartbeat_timestamp=TEST_NOW,
            ctx=ctx,
        )

        assert decision.action == "no_op"
        assert decision.reason is not None
        reason_lower = decision.reason.lower()
        assert (
            "terminal" in reason_lower
            or "expired" in reason_lower
            or "rejected" in reason_lower
        ), (
            f"Expected no_op reason to mention terminal/expired/rejected, got: {decision.reason!r}"
        )

    def test_no_intents_emitted_for_terminal_state(self) -> None:
        """No UPDATE intents must be emitted for a terminal-state heartbeat.

        If intents are emitted, handler_node_heartbeat will apply them to the
        database, erroneously updating liveness_deadline for a dead node.

        Currently FAILS: decide_heartbeat emits a postgres UPDATE intent.
        """
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.LIVENESS_EXPIRED,
            entity_id=node_id,
        )
        ctx = make_ctx()

        decision = service.decide_heartbeat(
            projection=projection,
            node_id=node_id,
            heartbeat_timestamp=TEST_NOW,
            ctx=ctx,
        )

        assert decision.action == "no_op"
        assert len(decision.intents) == 0, (
            f"Expected no intents for terminal-state no_op, got {len(decision.intents)} intent(s). "
            "Terminal-state heartbeats must not produce database side-effects."
        )

    def test_all_terminal_states_return_no_op(self) -> None:
        """Parametric: all is_terminal() states must produce no_op from decide_heartbeat."""
        service = RegistrationReducerService()
        terminal_states = [s for s in EnumRegistrationState if s.is_terminal()]

        assert len(terminal_states) >= 2, (
            "Expected at least LIVENESS_EXPIRED and REJECTED to be terminal states"
        )

        for state in terminal_states:
            node_id = uuid4()
            projection = make_projection(state, entity_id=node_id)
            ctx = make_ctx()

            decision = service.decide_heartbeat(
                projection=projection,
                node_id=node_id,
                heartbeat_timestamp=TEST_NOW,
                ctx=ctx,
            )

            assert decision.action == "no_op", (
                f"Expected no_op for terminal state {state!r}, "
                f"got action={decision.action!r}. "
                "All terminal states must be guarded in decide_heartbeat (OMN-4822)."
            )


# ---------------------------------------------------------------------------
# Regression: non-terminal states still get heartbeat processed
# ---------------------------------------------------------------------------


class TestDecideHeartbeatNonTerminalStillProcessed:
    """Non-terminal states must still have heartbeats processed normally.

    These tests verify the guard does not accidentally block valid heartbeats.
    They should be GREEN even before OMN-4822, since the current implementation
    already processes ACTIVE heartbeats correctly.
    """

    def test_active_node_heartbeat_returns_emit(self) -> None:
        """ACTIVE node heartbeat must still return action='emit' with UPDATE intent."""
        service = RegistrationReducerService()
        node_id = uuid4()
        projection = make_projection(
            EnumRegistrationState.ACTIVE,
            entity_id=node_id,
            liveness_deadline=TEST_NOW + timedelta(seconds=30),
        )
        ctx = make_ctx()

        decision = service.decide_heartbeat(
            projection=projection,
            node_id=node_id,
            heartbeat_timestamp=TEST_NOW,
            ctx=ctx,
        )

        assert decision.action == "emit", (
            f"Expected emit for ACTIVE state, got {decision.action!r}"
        )
        assert len(decision.intents) == 1, (
            f"Expected 1 UPDATE intent for active heartbeat, got {len(decision.intents)}"
        )
