# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""RED tests for ARCH-003: No Workflow FSM in Orchestrators.

These tests should FAIL initially (RED phase of TDD).
They verify that orchestrators with workflow FSM logic are detected.

KEY CLARIFICATION from ticket OMN-1099:
    - Reducers MAY implement aggregate state machines (that's their purpose)
    - Orchestrators must NOT implement workflow FSMs duplicating reducer transitions
    - Orchestrators are "reaction planners", not state machine owners

Rule ARCH-003 Scope:
    This rule targets orchestrator code that:
    1. Defines state transition tables (STATES, TRANSITIONS class variables)
    2. Tracks workflow state internally (_state, _workflow_state, _current_step)
    3. Implements transition methods (transition, can_transition, apply_transition)

    This rule does NOT apply to:
    1. Reducers (which legitimately own state machines)
    2. Orchestrators without FSM logic (pure coordination)
    3. Orchestrators that delegate state to reducers

Running Tests:
    # Run all ARCH-003 tests:
    pytest tests/unit/nodes/architecture_validator/test_validator_no_orchestrator_fsm.py -v

    # Run specific test:
    pytest tests/unit/nodes/architecture_validator/test_validator_no_orchestrator_fsm.py::TestNoOrchestratorFSM::test_detects_state_machine_in_orchestrator -v

Expected Outcome (RED Phase):
    These tests should FAIL because the stub validator always returns valid=True.
    The GREEN phase will implement actual FSM detection logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.nodes.architecture_validator.models.model_validation_result import (
    ModelArchitectureValidationResult,
)
from omnibase_infra.nodes.architecture_validator.validators.validator_no_orchestrator_fsm import (
    validate_no_orchestrator_fsm,
)

# =============================================================================
# TestNoOrchestratorFSM - Core Rule Tests
# =============================================================================


class TestNoOrchestratorFSM:
    """Tests for ARCH-003 rule: No Workflow FSM in Orchestrators.

    These tests verify that the validator correctly detects FSM patterns
    in orchestrator code that should be handled by reducers instead.

    The key distinction:
        - Reducers OWN state machines (STATES, TRANSITIONS, reduce())
        - Orchestrators COORDINATE work, delegating state to reducers
    """

    def test_detects_state_machine_in_orchestrator(self, tmp_path: Path) -> None:
        """Orchestrator with state machine should raise violation.

        This test verifies detection of the most explicit FSM anti-pattern:
        an orchestrator with STATES/TRANSITIONS class variables and
        internal state tracking.

        Expected Behavior:
            Validator should return valid=False with at least one ARCH-003 violation.
        """
        bad_code = '''
class OrchestratorOrderWorkflow(NodeOrchestrator):
    """BAD: Orchestrator implementing FSM logic."""

    STATES = ["pending", "processing", "completed", "failed"]
    TRANSITIONS = {
        "pending": ["processing"],
        "processing": ["completed", "failed"],
    }

    def __init__(self, container):
        super().__init__(container)
        self._state = "pending"  # VIOLATION: FSM state tracking

    def transition(self, new_state):  # VIOLATION: FSM transition logic
        if new_state in self.TRANSITIONS.get(self._state, []):
            self._state = new_state
'''
        test_file = tmp_path / "orchestrator_order.py"
        test_file.write_text(bad_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert not result.valid, (
            "Orchestrator with state machine should be detected as invalid"
        )
        assert len(result.violations) >= 1, (
            "At least one violation should be reported for FSM in orchestrator"
        )
        assert result.violations[0].rule_id == "ARCH-003", (
            f"Violation should be ARCH-003, got {result.violations[0].rule_id}"
        )
        # Check that the message mentions FSM or state machine
        message_lower = result.violations[0].message.lower()
        assert "fsm" in message_lower or "state machine" in message_lower, (
            f"Violation message should mention FSM or state machine, "
            f"got: {result.violations[0].message}"
        )

    def test_detects_workflow_state_tracking(self, tmp_path: Path) -> None:
        """Orchestrator tracking workflow state should raise violation.

        This test verifies detection of orchestrators that track workflow
        steps or state internally, rather than delegating to reducers.

        Expected Behavior:
            Validator should detect _workflow_state or _current_step patterns
            as violations of ARCH-003.
        """
        bad_code = '''
class OrchestratorPayment(NodeOrchestrator):
    """BAD: Orchestrator tracking workflow state internally."""

    def __init__(self, container):
        super().__init__(container)
        self._workflow_state = {}  # VIOLATION: workflow state tracking
        self._current_step = 0     # VIOLATION: step tracking

    def orchestrate(self, event):
        if self._current_step == 0:
            self._current_step = 1  # VIOLATION: state mutation
            return self.step_one(event)
        elif self._current_step == 1:
            self._current_step = 2
            return self.step_two(event)
'''
        test_file = tmp_path / "orchestrator_payment.py"
        test_file.write_text(bad_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert not result.valid, (
            "Orchestrator with workflow state tracking should be detected as invalid"
        )

    def test_detects_transition_methods(self, tmp_path: Path) -> None:
        """Orchestrator with transition/state methods should raise violation.

        This test verifies detection of method names that indicate FSM logic:
        - can_transition, apply_transition, do_transition
        - get_current_state, set_state
        - validate_transition

        Expected Behavior:
            Validator should detect FSM-related method patterns in orchestrators.
        """
        bad_code = '''
class OrchestratorShipping(NodeOrchestrator):
    """BAD: Orchestrator with FSM transition methods."""

    def can_transition(self, from_state, to_state):  # VIOLATION
        """Check if transition is valid."""
        pass

    def apply_transition(self, transition):  # VIOLATION
        """Apply a state transition."""
        pass

    def get_current_state(self):  # VIOLATION
        """Get current FSM state."""
        return self._state
'''
        test_file = tmp_path / "orchestrator_shipping.py"
        test_file.write_text(bad_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert not result.valid, (
            "Orchestrator with transition methods should be detected as invalid"
        )

    def test_detects_state_constants(self, tmp_path: Path) -> None:
        """Orchestrator with FSM state constants should raise violation.

        This test verifies detection of class-level state constants that
        indicate FSM ownership (e.g., STATE_PENDING, ALLOWED_TRANSITIONS).

        Expected Behavior:
            Validator should detect FSM-related constant patterns.
        """
        bad_code = '''
class OrchestratorApproval(NodeOrchestrator):
    """BAD: Orchestrator with FSM state constants."""

    STATE_PENDING = "pending"
    STATE_APPROVED = "approved"
    STATE_REJECTED = "rejected"

    ALLOWED_TRANSITIONS = {
        STATE_PENDING: [STATE_APPROVED, STATE_REJECTED],
    }

    def __init__(self, container):
        super().__init__(container)
        self._current_state = self.STATE_PENDING  # VIOLATION
'''
        test_file = tmp_path / "orchestrator_approval.py"
        test_file.write_text(bad_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert not result.valid, (
            "Orchestrator with FSM state constants should be detected as invalid"
        )


# =============================================================================
# TestOrchestratorFSMAllowedPatterns - Valid Code Tests
# =============================================================================


class TestOrchestratorFSMAllowedPatterns:
    """Tests verifying that valid orchestrator patterns pass ARCH-003.

    These tests ensure the validator does not produce false positives for:
    - Reducers with FSM logic (correct pattern)
    - Orchestrators without FSM logic (pure coordination)
    - Orchestrators that delegate state to reducers
    """

    def test_allows_reducer_state_machine(self, tmp_path: Path) -> None:
        """Reducers ARE allowed to have state machines.

        This is the key clarification from ticket OMN-1099:
        Reducers MAY implement aggregate state machines - that's their purpose.

        Expected Behavior:
            Validator should return valid=True for reducer FSM patterns.
        """
        good_code = '''
class ReducerOrderAggregate(NodeReducer):
    """GOOD: Reducer with FSM is CORRECT - this is what reducers do."""

    STATES = ["created", "processing", "completed"]
    TRANSITIONS = {"created": ["processing"], "processing": ["completed"]}

    def __init__(self, container):
        super().__init__(container)
        self._state = "created"  # OK for reducers

    def reduce(self, state, event):
        """Apply event to state and return intents."""
        if self.can_transition(event.target_state):
            self._state = event.target_state
        return self.get_intents()

    def can_transition(self, target_state):
        """Check if transition is valid - OK for reducers."""
        return target_state in self.TRANSITIONS.get(self._state, [])
'''
        test_file = tmp_path / "reducer_order.py"
        test_file.write_text(good_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert result.valid, (
            f"Reducers are EXEMPT from ARCH-003 - they OWN state machines. "
            f"Got violations: {result.violations}"
        )

    def test_allows_orchestrator_without_fsm(self, tmp_path: Path) -> None:
        """Orchestrator as reaction planner (no FSM) should pass.

        This test verifies that pure coordination orchestrators that
        delegate all state management to reducers pass validation.

        Expected Behavior:
            Validator should return valid=True for pure coordination.
        """
        good_code = '''
class OrchestratorUserOnboarding(NodeOrchestrator):
    """GOOD: Orchestrator as reaction planner - delegates state to reducers."""

    def __init__(self, container):
        super().__init__(container)
        self._reducer = container.resolve(ReducerUser)

    def orchestrate(self, event):
        """React to event, delegate state management to reducer."""
        # Get current state from reducer (not owned by orchestrator)
        state = self._reducer.get_projection()

        # Delegate state transition to reducer
        intents = self._reducer.reduce(state, event)

        # Plan reactions based on reducer output
        return self.plan_reactions(intents)

    def plan_reactions(self, intents):
        """Pure coordination - no state ownership."""
        return [self.execute_intent(i) for i in intents]
'''
        test_file = tmp_path / "orchestrator_user.py"
        test_file.write_text(good_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert result.valid, (
            f"Orchestrator without FSM should pass validation. "
            f"Got violations: {result.violations}"
        )

    def test_allows_orchestrator_workflow_coordination(self, tmp_path: Path) -> None:
        """Orchestrator coordinating workflow steps (without owning state) is OK.

        This test verifies that event-driven coordination patterns pass.
        The orchestrator reacts to events but does not track or own state.

        Expected Behavior:
            Validator should return valid=True for event-driven coordination.
        """
        good_code = '''
class OrchestratorCheckout(NodeOrchestrator):
    """GOOD: Event-driven coordination without state ownership."""

    def orchestrate(self, event):
        """Coordinate based on event type - no internal state tracking."""
        match event:
            case CartSubmitted():
                return [ValidateCart(), CalculateTotal()]
            case CartValidated():
                return [ProcessPayment()]
            case PaymentProcessed():
                return [CreateOrder(), SendConfirmation()]
            case PaymentFailed():
                return [NotifyUser(), ReleaseInventory()]
'''
        test_file = tmp_path / "orchestrator_checkout.py"
        test_file.write_text(good_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert result.valid, (
            f"Event-driven coordination should pass validation. "
            f"Got violations: {result.violations}"
        )

    def test_allows_orchestrator_with_correlation_tracking(
        self, tmp_path: Path
    ) -> None:
        """Orchestrator tracking correlation IDs (not state) should pass.

        This test ensures that legitimate tracking (correlation IDs, request
        metadata) is not confused with FSM state tracking.

        Expected Behavior:
            Validator should return valid=True for correlation tracking.
        """
        good_code = '''
class OrchestratorAsync(NodeOrchestrator):
    """GOOD: Correlation tracking is not FSM state tracking."""

    def __init__(self, container):
        super().__init__(container)
        self._pending_requests = {}  # Correlation tracking, not FSM state

    def orchestrate(self, event):
        """Track correlation for async coordination."""
        correlation_id = event.metadata.correlation_id

        if isinstance(event, RequestReceived):
            # Track for later correlation
            self._pending_requests[correlation_id] = event.timestamp
            return [self.dispatch_work(event)]

        if isinstance(event, WorkCompleted):
            # Correlate response with original request
            original_timestamp = self._pending_requests.pop(correlation_id, None)
            return [self.complete_request(event, original_timestamp)]
'''
        test_file = tmp_path / "orchestrator_async.py"
        test_file.write_text(good_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert result.valid, (
            f"Correlation tracking should pass validation. "
            f"Got violations: {result.violations}"
        )


# =============================================================================
# TestValidatorMetadata - Result Metadata Tests
# =============================================================================


class TestValidatorMetadata:
    """Tests for validator result metadata and formatting.

    These tests verify that the validator properly populates result metadata
    including files_checked, rules_checked, and violation details.
    """

    def test_result_includes_rule_id(self, tmp_path: Path) -> None:
        """Validation result should include ARCH-003 in rules_checked."""
        good_code = '''
class OrchestratorSimple(NodeOrchestrator):
    """Simple orchestrator with no FSM."""
    pass
'''
        test_file = tmp_path / "orchestrator_simple.py"
        test_file.write_text(good_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        assert "ARCH-003" in result.rules_checked, (
            f"rules_checked should include ARCH-003, got: {result.rules_checked}"
        )

    def test_result_includes_file_count(self, tmp_path: Path) -> None:
        """Validation result should include files_checked count."""
        test_file = tmp_path / "orchestrator_test.py"
        test_file.write_text("class Test: pass")

        result = validate_no_orchestrator_fsm(str(test_file))

        assert result.files_checked >= 1, (
            f"files_checked should be at least 1, got: {result.files_checked}"
        )

    def test_violation_includes_file_path(self, tmp_path: Path) -> None:
        """Violations should include the file path where issue was found."""
        bad_code = """
class OrchestratorBad(NodeOrchestrator):
    STATES = ["a", "b"]  # VIOLATION
    _state = "a"
"""
        test_file = tmp_path / "orchestrator_bad.py"
        test_file.write_text(bad_code)

        result = validate_no_orchestrator_fsm(str(test_file))

        # When validation fails, check violation details
        if not result.valid and result.violations:
            assert (
                str(test_file) in result.violations[0].location
                or test_file.name in result.violations[0].location
            ), (
                f"Violation location should reference test file, "
                f"got: {result.violations[0].location}"
            )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestNoOrchestratorFSM",
    "TestOrchestratorFSMAllowedPatterns",
    "TestValidatorMetadata",
]
