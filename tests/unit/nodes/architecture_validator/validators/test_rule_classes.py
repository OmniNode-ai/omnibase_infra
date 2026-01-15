# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for protocol-compliant Rule classes.

This module provides comprehensive test coverage for the three Rule classes
introduced in PR #124:
    - RuleNoDirectDispatch (ARCH-001 wrapper)
    - RuleNoHandlerPublishing (ARCH-002 wrapper)
    - RuleNoOrchestratorFSM (ARCH-003 wrapper)

Test Coverage:
    1. Property accessors (rule_id, name, description, severity)
    2. check() method graceful handling (non-string targets, nonexistent files)
    3. check() method violation detection (temp files with/without violations)

Related:
    - Ticket: OMN-1099 (Architecture Validator)
    - PR: #124 (Protocol-compliant Rule classes)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.nodes.architecture_validator.validators import (
    RuleNoDirectDispatch,
    RuleNoHandlerPublishing,
    RuleNoOrchestratorFSM,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rule_no_direct_dispatch() -> RuleNoDirectDispatch:
    """Create a RuleNoDirectDispatch instance for testing."""
    return RuleNoDirectDispatch()


@pytest.fixture
def rule_no_handler_publishing() -> RuleNoHandlerPublishing:
    """Create a RuleNoHandlerPublishing instance for testing."""
    return RuleNoHandlerPublishing()


@pytest.fixture
def rule_no_orchestrator_fsm() -> RuleNoOrchestratorFSM:
    """Create a RuleNoOrchestratorFSM instance for testing."""
    return RuleNoOrchestratorFSM()


@pytest.fixture
def create_temp_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary Python files.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture.

    Returns:
        A callable that takes (filename, content) and returns the Path.
    """

    def _create(filename: str, content: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create


# =============================================================================
# Test Class: RuleNoDirectDispatch
# =============================================================================


class TestRuleNoDirectDispatchProperties:
    """Tests for RuleNoDirectDispatch property accessors."""

    def test_rule_id_matches_contract(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Verify rule_id returns the canonical contract ID.

        The rule_id must match the ID defined in contract.yaml for
        proper integration with NodeArchitectureValidatorCompute.
        """
        assert rule_no_direct_dispatch.rule_id == "ARCH-001"

    def test_name_returns_human_readable_string(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Verify name returns a human-readable rule name."""
        name = rule_no_direct_dispatch.name
        assert isinstance(name, str)
        assert len(name) > 0
        assert name == "No Direct Handler Dispatch"

    def test_description_is_non_empty(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Verify description returns meaningful text."""
        description = rule_no_direct_dispatch.description
        assert isinstance(description, str)
        assert len(description) > 10  # Meaningful description expected
        # Should mention key concepts
        assert "runtime" in description.lower() or "dispatch" in description.lower()

    def test_severity_returns_warning(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Verify severity returns WARNING (non-blocking per contract).

        Contract specifies WARNING for ARCH-001 as direct dispatch
        is a code smell but may have legitimate use cases.
        """
        assert rule_no_direct_dispatch.severity == EnumValidationSeverity.WARNING


class TestRuleNoDirectDispatchCheckGraceful:
    """Tests for RuleNoDirectDispatch.check() graceful handling."""

    def test_check_none_target_passes(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Non-string target (None) should return passed=True.

        Rule only applies to file paths; other targets are not applicable.
        """
        result = rule_no_direct_dispatch.check(None)
        assert result.passed is True
        assert result.rule_id == "ARCH-001"

    def test_check_int_target_passes(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Non-string target (int) should return passed=True."""
        result = rule_no_direct_dispatch.check(42)
        assert result.passed is True

    def test_check_object_target_passes(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Non-string target (object) should return passed=True."""
        result = rule_no_direct_dispatch.check(object())
        assert result.passed is True

    def test_check_list_target_passes(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Non-string target (list) should return passed=True."""
        result = rule_no_direct_dispatch.check(["file.py"])
        assert result.passed is True

    def test_check_nonexistent_file_passes(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """Nonexistent file path should return passed=True.

        Missing files cannot be validated; graceful handling required.
        """
        result = rule_no_direct_dispatch.check("/nonexistent/path/to/file.py")
        assert result.passed is True


class TestRuleNoDirectDispatchCheckViolation:
    """Tests for RuleNoDirectDispatch.check() violation detection."""

    def test_check_violation_detected(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with direct dispatch violation should return passed=False."""
        violation_code = """
class SomeService:
    def process(self, event):
        handler = MyHandler(self.container)
        return handler.handle(event)  # VIOLATION
"""
        file_path = create_temp_file("violating_service.py", violation_code)
        result = rule_no_direct_dispatch.check(str(file_path))

        assert result.passed is False
        assert result.rule_id == "ARCH-001"

    def test_check_clean_file_passes(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File without violations should return passed=True."""
        clean_code = """
class CleanOrchestrator:
    def process(self, event):
        # Correct: dispatch through runtime
        return self.runtime.dispatch(event)
"""
        file_path = create_temp_file("clean_orchestrator.py", clean_code)
        result = rule_no_direct_dispatch.check(str(file_path))

        assert result.passed is True
        assert result.rule_id == "ARCH-001"

    def test_check_result_contains_details_on_failure(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Failed check should populate message and details."""
        violation_code = """
class BadService:
    def process(self, event):
        handler = SomeHandler(self.container)
        handler.handle(event)
"""
        file_path = create_temp_file("bad_service.py", violation_code)
        result = rule_no_direct_dispatch.check(str(file_path))

        assert result.passed is False
        assert result.message is not None
        assert len(result.message) > 0
        assert result.details is not None
        # Should contain violation details
        assert "target_name" in result.details or "location" in result.details

    def test_check_test_file_exempt(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Test files should be exempt from this rule."""
        test_code = """
def test_handler():
    handler = MyHandler(mock_container)
    result = handler.handle(test_event)
    assert result.success
"""
        file_path = create_temp_file("test_handler.py", test_code)
        result = rule_no_direct_dispatch.check(str(file_path))

        assert result.passed is True


# =============================================================================
# Test Class: RuleNoHandlerPublishing
# =============================================================================


class TestRuleNoHandlerPublishingProperties:
    """Tests for RuleNoHandlerPublishing property accessors."""

    def test_rule_id_matches_contract(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Verify rule_id returns the canonical contract ID."""
        assert rule_no_handler_publishing.rule_id == "ARCH-002"

    def test_name_returns_human_readable_string(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Verify name returns a human-readable rule name."""
        name = rule_no_handler_publishing.name
        assert isinstance(name, str)
        assert len(name) > 0
        assert name == "No Handler Publishing"

    def test_description_is_non_empty(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Verify description returns meaningful text."""
        description = rule_no_handler_publishing.description
        assert isinstance(description, str)
        assert len(description) > 10
        # Should mention handlers and publishing
        assert "handler" in description.lower()
        assert "publish" in description.lower() or "event" in description.lower()

    def test_severity_returns_error(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Verify severity returns ERROR (blocking violation).

        Handlers with direct event bus access is a serious architectural
        violation that should block deployment.
        """
        assert rule_no_handler_publishing.severity == EnumValidationSeverity.ERROR


class TestRuleNoHandlerPublishingCheckGraceful:
    """Tests for RuleNoHandlerPublishing.check() graceful handling."""

    def test_check_none_target_passes(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Non-string target (None) should return passed=True."""
        result = rule_no_handler_publishing.check(None)
        assert result.passed is True
        assert result.rule_id == "ARCH-002"

    def test_check_int_target_passes(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Non-string target (int) should return passed=True."""
        result = rule_no_handler_publishing.check(123)
        assert result.passed is True

    def test_check_object_target_passes(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Non-string target (object) should return passed=True."""
        result = rule_no_handler_publishing.check(object())
        assert result.passed is True

    def test_check_dict_target_passes(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Non-string target (dict) should return passed=True."""
        result = rule_no_handler_publishing.check({"file": "test.py"})
        assert result.passed is True

    def test_check_nonexistent_file_passes(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """Nonexistent file path should return passed=True."""
        result = rule_no_handler_publishing.check("/does/not/exist/handler.py")
        assert result.passed is True


class TestRuleNoHandlerPublishingCheckViolation:
    """Tests for RuleNoHandlerPublishing.check() violation detection."""

    def test_check_violation_bus_parameter(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Handler with event_bus in __init__ should return passed=False."""
        violation_code = """
class HandlerBad:
    def __init__(self, container, event_bus):
        self._bus = event_bus

    def handle(self, event):
        self._bus.publish(SomeEvent())
"""
        file_path = create_temp_file("handler_bad.py", violation_code)
        result = rule_no_handler_publishing.check(str(file_path))

        assert result.passed is False
        assert result.rule_id == "ARCH-002"

    def test_check_violation_publish_call(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Handler calling publish() should return passed=False."""
        violation_code = """
class HandlerWithPublish:
    def handle(self, event):
        self.publish(SomeEvent())  # VIOLATION
"""
        file_path = create_temp_file("handler_publish.py", violation_code)
        result = rule_no_handler_publishing.check(str(file_path))

        assert result.passed is False

    def test_check_clean_file_passes(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Handler without publish access should return passed=True."""
        clean_code = """
class HandlerGood:
    def __init__(self, container):
        self._container = container

    def handle(self, event):
        # Returns event for orchestrator to publish
        return SomeEvent(data=event.data)
"""
        file_path = create_temp_file("handler_good.py", clean_code)
        result = rule_no_handler_publishing.check(str(file_path))

        assert result.passed is True

    def test_check_result_contains_details_on_failure(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Failed check should populate message and details."""
        violation_code = """
class HandlerBadDetails:
    def __init__(self, container, publisher):
        self._publisher = publisher

    def handle(self, event):
        self._publisher.emit(SomeEvent())
"""
        file_path = create_temp_file("handler_bad_details.py", violation_code)
        result = rule_no_handler_publishing.check(str(file_path))

        assert result.passed is False
        assert result.message is not None
        assert len(result.message) > 0
        assert result.details is not None
        assert "target_name" in result.details or "suggestion" in result.details

    def test_check_orchestrator_with_bus_passes(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Orchestrators with event bus access should pass (not handlers)."""
        orchestrator_code = """
class OrchestratorWithBus:
    def __init__(self, container, event_bus):
        self._bus = event_bus

    def orchestrate(self, event):
        # Orchestrators ARE allowed to publish
        self._bus.publish(SomeEvent())
"""
        file_path = create_temp_file("orchestrator_bus.py", orchestrator_code)
        result = rule_no_handler_publishing.check(str(file_path))

        assert result.passed is True


# =============================================================================
# Test Class: RuleNoOrchestratorFSM
# =============================================================================


class TestRuleNoOrchestratorFSMProperties:
    """Tests for RuleNoOrchestratorFSM property accessors."""

    def test_rule_id_matches_contract(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Verify rule_id returns the canonical contract ID."""
        assert rule_no_orchestrator_fsm.rule_id == "ARCH-003"

    def test_name_returns_human_readable_string(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Verify name returns a human-readable rule name."""
        name = rule_no_orchestrator_fsm.name
        assert isinstance(name, str)
        assert len(name) > 0
        assert name == "No Workflow FSM in Orchestrators"

    def test_description_is_non_empty(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Verify description returns meaningful text."""
        description = rule_no_orchestrator_fsm.description
        assert isinstance(description, str)
        assert len(description) > 10
        # Should mention orchestrators and FSM/state
        assert "orchestrator" in description.lower()
        assert "fsm" in description.lower() or "state" in description.lower()

    def test_severity_returns_error(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Verify severity returns ERROR (blocking violation).

        Orchestrators with FSM logic violate separation of concerns;
        state machines belong in reducers.
        """
        assert rule_no_orchestrator_fsm.severity == EnumValidationSeverity.ERROR


class TestRuleNoOrchestratorFSMCheckGraceful:
    """Tests for RuleNoOrchestratorFSM.check() graceful handling."""

    def test_check_none_target_passes(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Non-string target (None) should return passed=True."""
        result = rule_no_orchestrator_fsm.check(None)
        assert result.passed is True
        assert result.rule_id == "ARCH-003"

    def test_check_int_target_passes(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Non-string target (int) should return passed=True."""
        result = rule_no_orchestrator_fsm.check(999)
        assert result.passed is True

    def test_check_object_target_passes(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Non-string target (object) should return passed=True."""
        result = rule_no_orchestrator_fsm.check(object())
        assert result.passed is True

    def test_check_tuple_target_passes(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Non-string target (tuple) should return passed=True."""
        result = rule_no_orchestrator_fsm.check(("path", "to", "file"))
        assert result.passed is True

    def test_check_nonexistent_file_passes(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """Nonexistent file path should return passed=True."""
        result = rule_no_orchestrator_fsm.check("/path/to/nowhere/orchestrator.py")
        assert result.passed is True


class TestRuleNoOrchestratorFSMCheckViolation:
    """Tests for RuleNoOrchestratorFSM.check() violation detection."""

    def test_check_violation_states_constant(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Orchestrator with STATES constant should return passed=False."""
        violation_code = """
class OrchestratorOrder:
    STATES = ["pending", "processing", "completed"]

    def orchestrate(self, event):
        pass
"""
        file_path = create_temp_file("orchestrator_states.py", violation_code)
        result = rule_no_orchestrator_fsm.check(str(file_path))

        assert result.passed is False
        assert result.rule_id == "ARCH-003"

    def test_check_violation_transitions_constant(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Orchestrator with TRANSITIONS constant should return passed=False."""
        violation_code = """
class OrchestratorWorkflow:
    TRANSITIONS = {
        "pending": ["processing"],
        "processing": ["completed", "failed"],
    }

    def orchestrate(self, event):
        pass
"""
        file_path = create_temp_file("orchestrator_transitions.py", violation_code)
        result = rule_no_orchestrator_fsm.check(str(file_path))

        assert result.passed is False

    def test_check_violation_state_attribute(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Orchestrator with _state attribute should return passed=False."""
        violation_code = """
class OrchestratorPayment:
    def __init__(self, container):
        self._state = "initial"  # VIOLATION

    def orchestrate(self, event):
        pass
"""
        file_path = create_temp_file("orchestrator_state.py", violation_code)
        result = rule_no_orchestrator_fsm.check(str(file_path))

        assert result.passed is False

    def test_check_violation_transition_method(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Orchestrator with transition method should return passed=False."""
        violation_code = """
class OrchestratorShipping:
    def can_transition(self, from_state, to_state):  # VIOLATION
        return True

    def orchestrate(self, event):
        pass
"""
        file_path = create_temp_file("orchestrator_transition.py", violation_code)
        result = rule_no_orchestrator_fsm.check(str(file_path))

        assert result.passed is False

    def test_check_clean_file_passes(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Orchestrator without FSM patterns should return passed=True."""
        clean_code = """
class OrchestratorClean:
    def __init__(self, container):
        self._container = container
        self._reducer = container.get_reducer()

    def orchestrate(self, event):
        # Delegate state management to reducer
        intents = self._reducer.reduce(self.state, event)
        return self.plan_reactions(intents)
"""
        file_path = create_temp_file("orchestrator_clean.py", clean_code)
        result = rule_no_orchestrator_fsm.check(str(file_path))

        assert result.passed is True

    def test_check_result_contains_details_on_failure(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Failed check should populate message and details."""
        violation_code = """
class OrchestratorFSMBad:
    FSM = {"initial": ["running"]}  # VIOLATION

    def orchestrate(self, event):
        pass
"""
        file_path = create_temp_file("orchestrator_fsm_bad.py", violation_code)
        result = rule_no_orchestrator_fsm.check(str(file_path))

        assert result.passed is False
        assert result.message is not None
        assert len(result.message) > 0
        assert result.details is not None
        assert "target_name" in result.details or "location" in result.details

    def test_check_reducer_with_fsm_passes(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Reducers with FSM patterns should pass (they own state machines)."""
        reducer_code = """
class ReducerOrder:
    STATES = ["created", "processing", "completed"]  # OK for reducers
    TRANSITIONS = {"created": ["processing"]}

    def __init__(self, container):
        self._state = "created"

    def reduce(self, state, event):
        return self.can_transition(state, event.target_state)
"""
        file_path = create_temp_file("reducer_order.py", reducer_code)
        result = rule_no_orchestrator_fsm.check(str(file_path))

        assert result.passed is True


# =============================================================================
# Cross-Rule Tests
# =============================================================================


class TestRuleClassesStateless:
    """Tests verifying rule classes are stateless and thread-safe."""

    def test_no_direct_dispatch_multiple_checks_independent(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Multiple checks on same rule instance should be independent."""
        violation_code = """
class Service:
    def process(self):
        handler = MyHandler()
        handler.handle(event)
"""
        clean_code = """
class CleanService:
    def process(self):
        self.runtime.dispatch(event)
"""
        violation_file = create_temp_file("violation.py", violation_code)
        clean_file = create_temp_file("clean.py", clean_code)

        # Check violation first
        result1 = rule_no_direct_dispatch.check(str(violation_file))
        assert result1.passed is False

        # Then check clean file - should not be affected by previous check
        result2 = rule_no_direct_dispatch.check(str(clean_file))
        assert result2.passed is True

        # Check violation again - should still fail
        result3 = rule_no_direct_dispatch.check(str(violation_file))
        assert result3.passed is False

    def test_rule_instances_independent(
        self, create_temp_file: Callable[[str, str], Path]
    ) -> None:
        """Different rule instances should not share state."""
        rule1 = RuleNoDirectDispatch()
        rule2 = RuleNoDirectDispatch()

        violation_code = """
class BadService:
    def process(self):
        handler = MyHandler()
        handler.handle(event)
"""
        file_path = create_temp_file("bad_service.py", violation_code)

        result1 = rule1.check(str(file_path))
        result2 = rule2.check(str(file_path))

        # Both should detect the violation independently
        assert result1.passed is False
        assert result2.passed is False
        # Results should be equal but not the same object
        assert result1.rule_id == result2.rule_id


class TestRuleClassesProtocolCompliance:
    """Tests verifying rule classes implement ProtocolArchitectureRule interface."""

    def test_no_direct_dispatch_has_all_properties(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """RuleNoDirectDispatch should have all required protocol properties."""
        # These should not raise AttributeError
        _ = rule_no_direct_dispatch.rule_id
        _ = rule_no_direct_dispatch.name
        _ = rule_no_direct_dispatch.description
        _ = rule_no_direct_dispatch.severity

    def test_no_handler_publishing_has_all_properties(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """RuleNoHandlerPublishing should have all required protocol properties."""
        _ = rule_no_handler_publishing.rule_id
        _ = rule_no_handler_publishing.name
        _ = rule_no_handler_publishing.description
        _ = rule_no_handler_publishing.severity

    def test_no_orchestrator_fsm_has_all_properties(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """RuleNoOrchestratorFSM should have all required protocol properties."""
        _ = rule_no_orchestrator_fsm.rule_id
        _ = rule_no_orchestrator_fsm.name
        _ = rule_no_orchestrator_fsm.description
        _ = rule_no_orchestrator_fsm.severity

    def test_all_rules_have_check_method(self) -> None:
        """All rule classes should have a check method accepting object."""
        for rule_class in [
            RuleNoDirectDispatch,
            RuleNoHandlerPublishing,
            RuleNoOrchestratorFSM,
        ]:
            rule = rule_class()
            assert hasattr(rule, "check")
            assert callable(rule.check)


class TestRuleCheckResultType:
    """Tests verifying check() returns proper ModelRuleCheckResult."""

    def test_no_direct_dispatch_returns_model_rule_check_result(
        self, rule_no_direct_dispatch: RuleNoDirectDispatch
    ) -> None:
        """check() should return ModelRuleCheckResult instance."""
        from omnibase_infra.nodes.architecture_validator.models import (
            ModelRuleCheckResult,
        )

        result = rule_no_direct_dispatch.check(None)
        assert isinstance(result, ModelRuleCheckResult)

    def test_no_handler_publishing_returns_model_rule_check_result(
        self, rule_no_handler_publishing: RuleNoHandlerPublishing
    ) -> None:
        """check() should return ModelRuleCheckResult instance."""
        from omnibase_infra.nodes.architecture_validator.models import (
            ModelRuleCheckResult,
        )

        result = rule_no_handler_publishing.check(None)
        assert isinstance(result, ModelRuleCheckResult)

    def test_no_orchestrator_fsm_returns_model_rule_check_result(
        self, rule_no_orchestrator_fsm: RuleNoOrchestratorFSM
    ) -> None:
        """check() should return ModelRuleCheckResult instance."""
        from omnibase_infra.nodes.architecture_validator.models import (
            ModelRuleCheckResult,
        )

        result = rule_no_orchestrator_fsm.check(None)
        assert isinstance(result, ModelRuleCheckResult)
