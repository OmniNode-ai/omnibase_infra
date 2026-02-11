# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for the validation adjudicator reducer node.

Tests:
- ModelAdjudicatorState FSM transitions (with_* methods)
- Guard methods (can_adjudicate, can_emit_verdict, can_reset)
- Duplicate event detection
- Invalid transition error handling
- ModelVerdict.from_state() scoring policy
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import (
    EnumAdjudicatorState,
    EnumCheckSeverity,
    EnumValidationVerdict,
)
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.validation.model_check_result import (
    ModelCheckResult,
)
from omnibase_infra.nodes.node_validation_adjudicator.models.model_adjudicator_state import (
    ModelAdjudicatorState,
)
from omnibase_infra.nodes.node_validation_adjudicator.models.model_verdict import (
    ModelVerdict,
)

pytestmark = pytest.mark.unit

# ============================================================================
# Helpers
# ============================================================================


def _make_check_result(
    check_code: str = "CHECK-TEST-001",
    severity: EnumCheckSeverity = EnumCheckSeverity.REQUIRED,
    passed: bool = True,
    skipped: bool = False,
    message: str = "",
) -> ModelCheckResult:
    """Create a check result for testing."""
    return ModelCheckResult(
        check_code=check_code,
        label=f"Check {check_code}",
        severity=severity,
        passed=passed,
        skipped=skipped,
        message=message,
        executed_at=datetime.now(tz=UTC),
    )


def _make_state(
    status: EnumAdjudicatorState = EnumAdjudicatorState.COLLECTING,
    check_results: tuple[ModelCheckResult, ...] = (),
) -> ModelAdjudicatorState:
    """Create an adjudicator state for testing."""
    return ModelAdjudicatorState(
        status=status,
        candidate_id=uuid4(),
        plan_id=uuid4(),
        check_results=check_results,
    )


# ============================================================================
# ModelAdjudicatorState -- Initial State
# ============================================================================


class TestAdjudicatorStateInitial:
    """Tests for ModelAdjudicatorState default construction."""

    def test_initial_status_is_collecting(self) -> None:
        """Default state is COLLECTING."""
        state = ModelAdjudicatorState()
        assert state.status == EnumAdjudicatorState.COLLECTING

    def test_initial_no_check_results(self) -> None:
        """Default state has empty check_results."""
        state = ModelAdjudicatorState()
        assert state.check_results == ()

    def test_initial_no_candidate(self) -> None:
        """Default state has no candidate_id."""
        state = ModelAdjudicatorState()
        assert state.candidate_id is None

    def test_initial_no_plan(self) -> None:
        """Default state has no plan_id."""
        state = ModelAdjudicatorState()
        assert state.plan_id is None

    def test_initial_no_last_event(self) -> None:
        """Default state has no last_processed_event_id."""
        state = ModelAdjudicatorState()
        assert state.last_processed_event_id is None


# ============================================================================
# ModelAdjudicatorState -- with_check_result()
# ============================================================================


class TestAdjudicatorStateWithCheckResult:
    """Tests for with_check_result() transition method."""

    def test_appends_result(self) -> None:
        """with_check_result appends a check result while staying in COLLECTING."""
        state = _make_state()
        result = _make_check_result()
        event_id = uuid4()

        new_state = state.with_check_result(result, event_id)

        assert len(new_state.check_results) == 1
        assert new_state.check_results[0] is result
        assert new_state.status == EnumAdjudicatorState.COLLECTING
        assert new_state.last_processed_event_id == event_id

    def test_appends_multiple_results(self) -> None:
        """Multiple with_check_result calls accumulate results."""
        state = _make_state()
        r1 = _make_check_result(check_code="C-1")
        r2 = _make_check_result(check_code="C-2")

        state = state.with_check_result(r1, uuid4())
        state = state.with_check_result(r2, uuid4())

        assert len(state.check_results) == 2
        assert state.check_results[0].check_code == "C-1"
        assert state.check_results[1].check_code == "C-2"

    def test_returns_new_instance(self) -> None:
        """with_check_result returns a new instance (defensive copy)."""
        state = _make_state()
        result = _make_check_result()

        new_state = state.with_check_result(result, uuid4())

        assert new_state is not state
        assert len(state.check_results) == 0  # original unchanged

    def test_raises_when_not_collecting(self) -> None:
        """Raises RuntimeHostError if state is not COLLECTING."""
        state = _make_state(status=EnumAdjudicatorState.ADJUDICATING)
        result = _make_check_result()

        with pytest.raises(RuntimeHostError):
            state.with_check_result(result, uuid4())


# ============================================================================
# ModelAdjudicatorState -- with_adjudication_started()
# ============================================================================


class TestAdjudicatorStateWithAdjudicationStarted:
    """Tests for with_adjudication_started() transition method."""

    def test_transitions_collecting_to_adjudicating(self) -> None:
        """Transitions from COLLECTING to ADJUDICATING."""
        state = _make_state()
        event_id = uuid4()

        new_state = state.with_adjudication_started(event_id)

        assert new_state.status == EnumAdjudicatorState.ADJUDICATING
        assert new_state.last_processed_event_id == event_id

    def test_preserves_check_results(self) -> None:
        """Check results are preserved during transition."""
        result = _make_check_result()
        state = _make_state(check_results=(result,))

        new_state = state.with_adjudication_started(uuid4())

        assert len(new_state.check_results) == 1
        assert new_state.check_results[0] is result

    def test_raises_when_already_adjudicating(self) -> None:
        """Raises RuntimeHostError if already in ADJUDICATING state."""
        state = _make_state(status=EnumAdjudicatorState.ADJUDICATING)

        with pytest.raises(RuntimeHostError):
            state.with_adjudication_started(uuid4())

    def test_raises_when_verdict_emitted(self) -> None:
        """Raises RuntimeHostError if in VERDICT_EMITTED state."""
        state = _make_state(status=EnumAdjudicatorState.VERDICT_EMITTED)

        with pytest.raises(RuntimeHostError):
            state.with_adjudication_started(uuid4())


# ============================================================================
# ModelAdjudicatorState -- with_verdict_emitted()
# ============================================================================


class TestAdjudicatorStateWithVerdictEmitted:
    """Tests for with_verdict_emitted() transition method."""

    def test_transitions_adjudicating_to_verdict_emitted(self) -> None:
        """Transitions from ADJUDICATING to VERDICT_EMITTED."""
        state = _make_state(status=EnumAdjudicatorState.ADJUDICATING)
        event_id = uuid4()

        new_state = state.with_verdict_emitted(event_id)

        assert new_state.status == EnumAdjudicatorState.VERDICT_EMITTED
        assert new_state.last_processed_event_id == event_id

    def test_raises_when_collecting(self) -> None:
        """Raises RuntimeHostError if in COLLECTING state (skips adjudicating)."""
        state = _make_state(status=EnumAdjudicatorState.COLLECTING)

        with pytest.raises(RuntimeHostError):
            state.with_verdict_emitted(uuid4())

    def test_raises_when_already_verdict_emitted(self) -> None:
        """Raises RuntimeHostError if already in VERDICT_EMITTED state."""
        state = _make_state(status=EnumAdjudicatorState.VERDICT_EMITTED)

        with pytest.raises(RuntimeHostError):
            state.with_verdict_emitted(uuid4())


# ============================================================================
# ModelAdjudicatorState -- with_reset()
# ============================================================================


class TestAdjudicatorStateWithReset:
    """Tests for with_reset() transition method."""

    def test_transitions_verdict_emitted_to_collecting(self) -> None:
        """Transitions from VERDICT_EMITTED to COLLECTING."""
        state = _make_state(status=EnumAdjudicatorState.VERDICT_EMITTED)
        event_id = uuid4()

        new_state = state.with_reset(event_id)

        assert new_state.status == EnumAdjudicatorState.COLLECTING
        assert new_state.last_processed_event_id == event_id

    def test_clears_data_on_reset(self) -> None:
        """Reset clears candidate_id, plan_id, and check_results."""
        result = _make_check_result()
        state = ModelAdjudicatorState(
            status=EnumAdjudicatorState.VERDICT_EMITTED,
            candidate_id=uuid4(),
            plan_id=uuid4(),
            check_results=(result,),
        )

        new_state = state.with_reset(uuid4())

        assert new_state.candidate_id is None
        assert new_state.plan_id is None
        assert new_state.check_results == ()

    def test_raises_when_collecting(self) -> None:
        """Raises RuntimeHostError if in COLLECTING state."""
        state = _make_state(status=EnumAdjudicatorState.COLLECTING)

        with pytest.raises(RuntimeHostError):
            state.with_reset(uuid4())

    def test_raises_when_adjudicating(self) -> None:
        """Raises RuntimeHostError if in ADJUDICATING state."""
        state = _make_state(status=EnumAdjudicatorState.ADJUDICATING)

        with pytest.raises(RuntimeHostError):
            state.with_reset(uuid4())


# ============================================================================
# ModelAdjudicatorState -- Guard Methods
# ============================================================================


class TestAdjudicatorStateGuards:
    """Tests for guard methods (can_adjudicate, can_emit_verdict, can_reset)."""

    def test_can_adjudicate_when_collecting(self) -> None:
        """can_adjudicate() returns True in COLLECTING state."""
        state = _make_state(status=EnumAdjudicatorState.COLLECTING)
        assert state.can_adjudicate() is True

    def test_cannot_adjudicate_when_adjudicating(self) -> None:
        """can_adjudicate() returns False in ADJUDICATING state."""
        state = _make_state(status=EnumAdjudicatorState.ADJUDICATING)
        assert state.can_adjudicate() is False

    def test_cannot_adjudicate_when_verdict_emitted(self) -> None:
        """can_adjudicate() returns False in VERDICT_EMITTED state."""
        state = _make_state(status=EnumAdjudicatorState.VERDICT_EMITTED)
        assert state.can_adjudicate() is False

    def test_can_emit_verdict_when_adjudicating(self) -> None:
        """can_emit_verdict() returns True in ADJUDICATING state."""
        state = _make_state(status=EnumAdjudicatorState.ADJUDICATING)
        assert state.can_emit_verdict() is True

    def test_cannot_emit_verdict_when_collecting(self) -> None:
        """can_emit_verdict() returns False in COLLECTING state."""
        state = _make_state(status=EnumAdjudicatorState.COLLECTING)
        assert state.can_emit_verdict() is False

    def test_cannot_emit_verdict_when_verdict_emitted(self) -> None:
        """can_emit_verdict() returns False in VERDICT_EMITTED state."""
        state = _make_state(status=EnumAdjudicatorState.VERDICT_EMITTED)
        assert state.can_emit_verdict() is False

    def test_can_reset_when_verdict_emitted(self) -> None:
        """can_reset() returns True in VERDICT_EMITTED state."""
        state = _make_state(status=EnumAdjudicatorState.VERDICT_EMITTED)
        assert state.can_reset() is True

    def test_cannot_reset_when_collecting(self) -> None:
        """can_reset() returns False in COLLECTING state."""
        state = _make_state(status=EnumAdjudicatorState.COLLECTING)
        assert state.can_reset() is False

    def test_cannot_reset_when_adjudicating(self) -> None:
        """can_reset() returns False in ADJUDICATING state."""
        state = _make_state(status=EnumAdjudicatorState.ADJUDICATING)
        assert state.can_reset() is False


# ============================================================================
# ModelAdjudicatorState -- Duplicate Event Detection
# ============================================================================


class TestAdjudicatorStateDuplicateEvent:
    """Tests for is_duplicate_event() idempotency check."""

    def test_detects_duplicate(self) -> None:
        """is_duplicate_event returns True for the same event_id."""
        event_id = uuid4()
        state = _make_state()
        new_state = state.with_check_result(_make_check_result(), event_id)

        assert new_state.is_duplicate_event(event_id) is True

    def test_does_not_flag_different_event(self) -> None:
        """is_duplicate_event returns False for a different event_id."""
        state = _make_state()
        new_state = state.with_check_result(_make_check_result(), uuid4())

        assert new_state.is_duplicate_event(uuid4()) is False

    def test_initial_state_no_duplicate(self) -> None:
        """Initial state with no last_processed_event_id never detects a duplicate."""
        state = ModelAdjudicatorState()
        assert state.is_duplicate_event(uuid4()) is False


# ============================================================================
# ModelVerdict.from_state()
# ============================================================================


class TestModelVerdictFromState:
    """Tests for ModelVerdict.from_state() scoring policy."""

    def test_all_required_pass_produces_pass_verdict(self) -> None:
        """All REQUIRED checks passing and score >= threshold -> PASS."""
        results = (
            _make_check_result(
                check_code="C-1", severity=EnumCheckSeverity.REQUIRED, passed=True
            ),
            _make_check_result(
                check_code="C-2", severity=EnumCheckSeverity.REQUIRED, passed=True
            ),
            _make_check_result(
                check_code="C-3", severity=EnumCheckSeverity.RECOMMENDED, passed=True
            ),
        )
        state = _make_state(check_results=results)

        verdict = ModelVerdict.from_state(state)

        assert verdict.verdict == EnumValidationVerdict.PASS
        assert verdict.score == 1.0
        assert verdict.blocking_failures == ()
        assert verdict.passed_checks == 3
        assert verdict.failed_checks == 0

    def test_required_failure_produces_fail_verdict(self) -> None:
        """A REQUIRED check failing -> FAIL with blocking_failures."""
        results = (
            _make_check_result(
                check_code="C-1", severity=EnumCheckSeverity.REQUIRED, passed=False
            ),
            _make_check_result(
                check_code="C-2", severity=EnumCheckSeverity.REQUIRED, passed=True
            ),
        )
        state = _make_state(check_results=results)

        verdict = ModelVerdict.from_state(state)

        assert verdict.verdict == EnumValidationVerdict.FAIL
        assert "C-1" in verdict.blocking_failures
        assert verdict.failed_checks == 1

    def test_score_below_threshold_produces_quarantine(self) -> None:
        """Score below threshold without blocking failures -> QUARANTINE."""
        # 1 passed, 2 failed recommended -> score 0.33 < 0.8
        results = (
            _make_check_result(
                check_code="C-1", severity=EnumCheckSeverity.RECOMMENDED, passed=True
            ),
            _make_check_result(
                check_code="C-2",
                severity=EnumCheckSeverity.RECOMMENDED,
                passed=False,
                message="flaky test",
            ),
            _make_check_result(
                check_code="C-3",
                severity=EnumCheckSeverity.RECOMMENDED,
                passed=False,
                message="diff too large",
            ),
        )
        state = _make_state(check_results=results)

        verdict = ModelVerdict.from_state(state, score_threshold=0.8)

        assert verdict.verdict == EnumValidationVerdict.QUARANTINE
        assert verdict.score < 0.8
        assert len(verdict.quarantine_reasons) > 0

    def test_score_calculation(self) -> None:
        """Score is passed / (passed + failed), skipping skipped checks."""
        results = (
            _make_check_result(check_code="C-1", passed=True),
            _make_check_result(check_code="C-2", passed=False, skipped=True),  # skipped
            _make_check_result(check_code="C-3", passed=True),
        )
        state = _make_state(check_results=results)

        verdict = ModelVerdict.from_state(state)

        # 2 passed, 0 failed (1 skipped doesn't count), so score = 2/2 = 1.0
        assert verdict.score == 1.0
        assert verdict.skipped_checks == 1
        assert verdict.total_checks == 3

    def test_all_skipped_produces_score_one(self) -> None:
        """If all checks are skipped, score defaults to 1.0."""
        results = (_make_check_result(check_code="C-1", passed=False, skipped=True),)
        state = _make_state(check_results=results)

        verdict = ModelVerdict.from_state(state)

        assert verdict.score == 1.0

    def test_verdict_references_state_ids(self) -> None:
        """Verdict's candidate_id and plan_id match the input state."""
        state = _make_state(check_results=())

        verdict = ModelVerdict.from_state(state)

        assert verdict.candidate_id == state.candidate_id
        assert verdict.plan_id == state.plan_id

    def test_verdict_has_adjudicated_at(self) -> None:
        """Verdict includes an adjudicated_at timestamp."""
        state = _make_state(check_results=())

        verdict = ModelVerdict.from_state(state)

        assert isinstance(verdict.adjudicated_at, datetime)

    def test_raises_when_candidate_id_none(self) -> None:
        """Raises RuntimeHostError if state has no candidate_id."""
        state = ModelAdjudicatorState(candidate_id=None, plan_id=uuid4())

        with pytest.raises(RuntimeHostError, match="candidate_id"):
            ModelVerdict.from_state(state)

    def test_raises_when_plan_id_none(self) -> None:
        """Raises RuntimeHostError if state has no plan_id."""
        state = ModelAdjudicatorState(candidate_id=uuid4(), plan_id=None)

        with pytest.raises(RuntimeHostError, match="plan_id"):
            ModelVerdict.from_state(state)

    def test_propagates_correlation_id_on_candidate_id_none(self) -> None:
        """Caller correlation_id is propagated into the error context when candidate_id is None."""
        caller_id = uuid4()
        state = ModelAdjudicatorState(candidate_id=None, plan_id=uuid4())

        with pytest.raises(RuntimeHostError) as exc_info:
            ModelVerdict.from_state(state, correlation_id=caller_id)

        assert exc_info.value.correlation_id == caller_id

    def test_propagates_correlation_id_on_plan_id_none(self) -> None:
        """Caller correlation_id is propagated into the error context when plan_id is None."""
        caller_id = uuid4()
        state = ModelAdjudicatorState(candidate_id=uuid4(), plan_id=None)

        with pytest.raises(RuntimeHostError) as exc_info:
            ModelVerdict.from_state(state, correlation_id=caller_id)

        assert exc_info.value.correlation_id == caller_id

    def test_auto_generates_correlation_id_when_not_provided(self) -> None:
        """When no correlation_id is passed, error context auto-generates one."""
        state = ModelAdjudicatorState(candidate_id=None, plan_id=uuid4())

        with pytest.raises(RuntimeHostError) as exc_info:
            ModelVerdict.from_state(state)

        assert exc_info.value.correlation_id is not None
        assert isinstance(exc_info.value.correlation_id, UUID)


# ============================================================================
# Full FSM Cycle
# ============================================================================


class TestAdjudicatorFullCycle:
    """Integration test: full FSM cycle from collecting through reset."""

    def test_full_cycle(self) -> None:
        """Walk through the entire FSM: collecting -> adjudicating -> verdict -> reset."""
        # Start in COLLECTING
        state = _make_state()
        assert state.status == EnumAdjudicatorState.COLLECTING

        # Add check results
        state = state.with_check_result(
            _make_check_result(check_code="C-1", passed=True), uuid4()
        )
        state = state.with_check_result(
            _make_check_result(check_code="C-2", passed=True), uuid4()
        )
        assert len(state.check_results) == 2

        # Transition to ADJUDICATING
        state = state.with_adjudication_started(uuid4())
        assert state.status == EnumAdjudicatorState.ADJUDICATING

        # Emit verdict
        state = state.with_verdict_emitted(uuid4())
        assert state.status == EnumAdjudicatorState.VERDICT_EMITTED

        # Reset for next run
        state = state.with_reset(uuid4())
        assert state.status == EnumAdjudicatorState.COLLECTING
        assert state.check_results == ()
        assert state.candidate_id is None
        assert state.plan_id is None
