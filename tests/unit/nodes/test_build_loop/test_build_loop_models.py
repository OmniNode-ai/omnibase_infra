# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for build loop Pydantic models.

Related:
    - OMN-7311, OMN-7312: Foundation models
    - OMN-7323: Canary integration test
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums.enum_build_loop_intent_type import EnumBuildLoopIntentType
from omnibase_infra.enums.enum_build_loop_phase import EnumBuildLoopPhase
from omnibase_infra.enums.enum_buildability import EnumBuildability
from omnibase_infra.nodes.node_loop_state_reducer.models.model_build_loop_event import (
    ModelBuildLoopEvent,
)
from omnibase_infra.nodes.node_loop_state_reducer.models.model_build_loop_intent import (
    ModelBuildLoopIntent,
)
from omnibase_infra.nodes.node_loop_state_reducer.models.model_build_loop_state import (
    ModelBuildLoopState,
)
from omnibase_infra.nodes.node_ticket_classify_compute.models.model_ticket_classification import (
    ModelTicketClassification,
    ModelTicketForClassification,
)
from omnibase_infra.nodes.node_rsd_fill_compute.models.model_rsd_fill import (
    ModelScoredTicket,
)


@pytest.mark.unit
class TestBuildLoopState:
    """Tests for ModelBuildLoopState."""

    def test_default_state(self):
        cid = uuid4()
        state = ModelBuildLoopState(correlation_id=cid)
        assert state.phase == EnumBuildLoopPhase.IDLE
        assert state.cycle_number == 0
        assert state.consecutive_failures == 0
        assert state.max_consecutive_failures == 3
        assert state.skip_closeout is False
        assert state.dry_run is False

    def test_frozen(self):
        state = ModelBuildLoopState(correlation_id=uuid4())
        with pytest.raises(ValidationError):
            state.phase = EnumBuildLoopPhase.BUILDING  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            ModelBuildLoopState(correlation_id=uuid4(), bogus="field")  # type: ignore[call-arg]


@pytest.mark.unit
class TestBuildLoopEvent:
    """Tests for ModelBuildLoopEvent."""

    def test_valid_event(self):
        evt = ModelBuildLoopEvent(
            correlation_id=uuid4(),
            source_phase=EnumBuildLoopPhase.CLOSING_OUT,
            success=True,
            timestamp=datetime.now(tz=UTC),
        )
        assert evt.success is True
        assert evt.error_message is None

    def test_failure_event(self):
        evt = ModelBuildLoopEvent(
            correlation_id=uuid4(),
            source_phase=EnumBuildLoopPhase.VERIFYING,
            success=False,
            timestamp=datetime.now(tz=UTC),
            error_message="Runtime health check failed",
        )
        assert evt.success is False
        assert "Runtime" in evt.error_message  # type: ignore[operator]


@pytest.mark.unit
class TestBuildLoopIntent:
    """Tests for ModelBuildLoopIntent."""

    def test_valid_intent(self):
        intent = ModelBuildLoopIntent(
            intent_type=EnumBuildLoopIntentType.START_VERIFY,
            correlation_id=uuid4(),
            cycle_number=1,
            from_phase=EnumBuildLoopPhase.VERIFYING,
        )
        assert intent.intent_type == EnumBuildLoopIntentType.START_VERIFY

    def test_payload_default(self):
        intent = ModelBuildLoopIntent(
            intent_type=EnumBuildLoopIntentType.START_BUILD,
            correlation_id=uuid4(),
            cycle_number=1,
            from_phase=EnumBuildLoopPhase.BUILDING,
        )
        assert intent.payload == {}


@pytest.mark.unit
class TestTicketClassification:
    """Tests for ModelTicketClassification and EnumBuildability."""

    def test_auto_buildable(self):
        tc = ModelTicketClassification(
            ticket_id="OMN-1234",
            title="Add build loop node",
            buildability=EnumBuildability.AUTO_BUILDABLE,
            confidence=0.8,
            matched_keywords=("add", "node"),
        )
        assert tc.buildability == EnumBuildability.AUTO_BUILDABLE

    def test_all_buildability_values(self):
        values = {e.value for e in EnumBuildability}
        assert values == {"auto_buildable", "needs_arch_decision", "blocked", "skip"}


@pytest.mark.unit
class TestScoredTicket:
    """Tests for ModelScoredTicket."""

    def test_valid_ticket(self):
        t = ModelScoredTicket(
            ticket_id="OMN-100",
            title="Test ticket",
            rsd_score=7.5,
            priority=2,
        )
        assert t.rsd_score == 7.5

    def test_negative_rsd_rejected(self):
        with pytest.raises(ValidationError):
            ModelScoredTicket(
                ticket_id="OMN-100",
                title="Test",
                rsd_score=-1.0,
            )
