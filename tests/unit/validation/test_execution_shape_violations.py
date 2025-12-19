# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for Execution Shape Violations (OMN-958).

This module contains the 5 required "known bad" test cases that validate
the execution shape validators correctly detect and reject handlers that
violate ONEX 4-node architecture constraints.

Test Cases:
    1. test_reducer_returning_events_rejected - Reducer cannot return EVENT
    2. test_orchestrator_performing_io_rejected - Orchestrator cannot return INTENT/PROJECTION
    3. test_effect_returning_projections_rejected - Effect cannot return PROJECTION
    4. test_reducer_accessing_system_time_rejected - Reducer cannot access time.time()/datetime.now()
    5. test_handler_direct_publish_rejected - All handlers forbidden from direct .publish()

Note:
    This module uses pytest's tmp_path fixture for temporary file management.
    The fixture automatically handles cleanup after each test, eliminating
    the need for manual try/finally blocks with file.unlink().
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.validation import (
    ExecutionShapeValidator,
    ExecutionShapeViolationError,
    RuntimeShapeValidator,
    enforce_execution_shape,
)


def _write_test_file(tmp_path: Path, code: str) -> Path:
    """Write test code to a temporary Python file.

    Helper function that creates a temporary .py file with the given code.
    The file is automatically cleaned up by pytest's tmp_path fixture.

    Args:
        tmp_path: pytest's tmp_path fixture providing a temp directory.
        code: Python source code to write to the file.

    Returns:
        Path to the created temporary file.
    """
    file_path = tmp_path / "test_handler.py"
    file_path.write_text(code)
    return file_path


class TestReducerReturningEventsRejected:
    """Test case 1: Reducer handler returning Event type must be rejected."""

    def test_reducer_returning_event_type_annotation_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Reducer with Event return type annotation detected by AST validator."""
        bad_code = textwrap.dedent("""
            class OrderCreatedEvent:
                def __init__(self, order_id: str):
                    self.order_id = order_id

            class OrderReducerHandler:
                def handle(self, command) -> OrderCreatedEvent:
                    return OrderCreatedEvent(order_id="123")
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        # Assert REDUCER_RETURNS_EVENTS violation found
        assert len(violations) >= 1
        reducer_event_violations = [
            v
            for v in violations
            if v.violation_type == EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS
        ]
        assert len(reducer_event_violations) >= 1
        assert reducer_event_violations[0].handler_type == EnumHandlerType.REDUCER
        assert "event" in reducer_event_violations[0].message.lower()

    def test_reducer_returning_event_call_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Reducer returning Event(...) call detected by AST validator."""
        bad_code = textwrap.dedent("""
            class OrderReducer:
                def reduce(self, state, action):
                    # Bad: reducer returning an event
                    return OrderCreatedEvent(order_id=action.order_id)
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        reducer_event_violations = [
            v
            for v in violations
            if v.violation_type == EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS
        ]
        assert len(reducer_event_violations) >= 1

    def test_reducer_returning_event_rejected_by_runtime(self) -> None:
        """Reducer returning Event rejected by runtime shape validator."""

        # Define a message class that the runtime validator can detect
        class OrderCreatedEvent:
            category = EnumMessageCategory.EVENT

        validator = RuntimeShapeValidator()

        # Should detect violation when reducer returns EVENT
        violation = validator.validate_handler_output(
            handler_type=EnumHandlerType.REDUCER,
            output=OrderCreatedEvent(),
            output_category=EnumMessageCategory.EVENT,
        )

        assert violation is not None
        assert (
            violation.violation_type
            == EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS
        )
        assert violation.handler_type == EnumHandlerType.REDUCER
        assert violation.severity == "error"

    def test_reducer_returning_event_raises_via_decorator(self) -> None:
        """Reducer decorated function raises ExecutionShapeViolationError for EVENT."""

        class OrderCreatedEvent:
            category = EnumMessageCategory.EVENT

        @enforce_execution_shape(EnumHandlerType.REDUCER)
        def bad_reducer_handler(data: dict) -> OrderCreatedEvent:
            return OrderCreatedEvent()

        with pytest.raises(ExecutionShapeViolationError) as exc_info:
            bad_reducer_handler({"order_id": "123"})

        assert (
            exc_info.value.violation.violation_type
            == EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS
        )


class TestOrchestratorPerformingIORejected:
    """Test case 2: Orchestrator handler returning Intent or Projection must be rejected."""

    def test_orchestrator_returning_intent_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Orchestrator returning Intent type detected by AST validator."""
        bad_code = textwrap.dedent("""
            class CheckoutIntent:
                pass

            class OrderOrchestratorHandler:
                def orchestrate(self, command) -> CheckoutIntent:
                    return CheckoutIntent()
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        # Assert ORCHESTRATOR_RETURNS_INTENTS violation found
        intent_violations = [
            v
            for v in violations
            if v.violation_type
            == EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_INTENTS
        ]
        assert len(intent_violations) >= 1
        assert intent_violations[0].handler_type == EnumHandlerType.ORCHESTRATOR

    def test_orchestrator_returning_projection_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Orchestrator returning Projection type detected by AST validator."""
        bad_code = textwrap.dedent("""
            class OrderSummaryProjection:
                pass

            class OrderOrchestrator:
                def handle(self, event) -> OrderSummaryProjection:
                    return OrderSummaryProjection()
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        # Assert ORCHESTRATOR_RETURNS_PROJECTIONS violation found
        projection_violations = [
            v
            for v in violations
            if v.violation_type
            == EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_PROJECTIONS
        ]
        assert len(projection_violations) >= 1
        assert projection_violations[0].handler_type == EnumHandlerType.ORCHESTRATOR

    def test_orchestrator_returning_intent_rejected_by_runtime(self) -> None:
        """Orchestrator returning Intent rejected by runtime validator."""

        class CheckoutIntent:
            category = EnumMessageCategory.INTENT

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            handler_type=EnumHandlerType.ORCHESTRATOR,
            output=CheckoutIntent(),
            output_category=EnumMessageCategory.INTENT,
        )

        assert violation is not None
        assert (
            violation.violation_type
            == EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_INTENTS
        )

    def test_orchestrator_returning_projection_rejected_by_runtime(self) -> None:
        """Orchestrator returning Projection rejected by runtime validator."""

        class OrderProjection:
            category = EnumMessageCategory.PROJECTION

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            handler_type=EnumHandlerType.ORCHESTRATOR,
            output=OrderProjection(),
            output_category=EnumMessageCategory.PROJECTION,
        )

        assert violation is not None
        assert (
            violation.violation_type
            == EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_PROJECTIONS
        )

    def test_orchestrator_returning_intent_raises_via_decorator(self) -> None:
        """Orchestrator decorated function raises for Intent return."""

        class CheckoutIntent:
            category = EnumMessageCategory.INTENT

        @enforce_execution_shape(EnumHandlerType.ORCHESTRATOR)
        def bad_orchestrator(data: dict) -> CheckoutIntent:
            return CheckoutIntent()

        with pytest.raises(ExecutionShapeViolationError) as exc_info:
            bad_orchestrator({"cart_id": "abc"})

        assert (
            exc_info.value.violation.violation_type
            == EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_INTENTS
        )

    def test_orchestrator_returning_projection_raises_via_decorator(self) -> None:
        """Orchestrator decorated function raises for Projection return."""

        class OrderProjection:
            category = EnumMessageCategory.PROJECTION

        @enforce_execution_shape(EnumHandlerType.ORCHESTRATOR)
        def bad_orchestrator(data: dict) -> OrderProjection:
            return OrderProjection()

        with pytest.raises(ExecutionShapeViolationError) as exc_info:
            bad_orchestrator({"order_id": "123"})

        assert (
            exc_info.value.violation.violation_type
            == EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_PROJECTIONS
        )


class TestEffectReturningProjectionsRejected:
    """Test case 3: Effect handler returning Projection type must be rejected."""

    def test_effect_returning_projection_type_annotation_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Effect with Projection return type annotation detected by AST validator."""
        bad_code = textwrap.dedent("""
            class UserProfileProjection:
                pass

            class UserEffectHandler:
                def handle(self, command) -> UserProfileProjection:
                    return UserProfileProjection()
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        # Assert EFFECT_RETURNS_PROJECTIONS violation found
        projection_violations = [
            v
            for v in violations
            if v.violation_type == EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
        ]
        assert len(projection_violations) >= 1
        assert projection_violations[0].handler_type == EnumHandlerType.EFFECT
        assert "projection" in projection_violations[0].message.lower()

    def test_effect_returning_projection_call_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Effect returning Projection(...) call detected by AST validator."""
        bad_code = textwrap.dedent("""
            class DatabaseEffect:
                def execute(self, query):
                    # Bad: effect returning a projection
                    return OrderSummaryProjection(total=100)
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        projection_violations = [
            v
            for v in violations
            if v.violation_type == EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
        ]
        assert len(projection_violations) >= 1

    def test_effect_returning_projection_rejected_by_runtime(self) -> None:
        """Effect returning Projection rejected by runtime shape validator."""

        class UserProfileProjection:
            category = EnumMessageCategory.PROJECTION

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            handler_type=EnumHandlerType.EFFECT,
            output=UserProfileProjection(),
            output_category=EnumMessageCategory.PROJECTION,
        )

        assert violation is not None
        assert (
            violation.violation_type
            == EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
        )
        assert violation.handler_type == EnumHandlerType.EFFECT
        assert violation.severity == "error"

    def test_effect_returning_projection_raises_via_decorator(self) -> None:
        """Effect decorated function raises ExecutionShapeViolationError for Projection."""

        class UserProfileProjection:
            category = EnumMessageCategory.PROJECTION

        @enforce_execution_shape(EnumHandlerType.EFFECT)
        def bad_effect_handler(data: dict) -> UserProfileProjection:
            return UserProfileProjection()

        with pytest.raises(ExecutionShapeViolationError) as exc_info:
            bad_effect_handler({"user_id": "456"})

        assert (
            exc_info.value.violation.violation_type
            == EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
        )


class TestReducerAccessingSystemTimeRejected:
    """Test case 4: Reducer handler accessing system time must be rejected."""

    def test_reducer_calling_time_time_rejected_by_ast(self, tmp_path: Path) -> None:
        """Reducer calling time.time() detected by AST validator."""
        bad_code = textwrap.dedent("""
            import time

            class OrderReducerHandler:
                def reduce(self, state, event):
                    # Bad: accessing non-deterministic system time
                    timestamp = time.time()
                    return {"updated_at": timestamp}
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        # Assert REDUCER_ACCESSES_SYSTEM_TIME violation found
        time_violations = [
            v
            for v in violations
            if v.violation_type
            == EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME
        ]
        assert len(time_violations) >= 1
        assert time_violations[0].handler_type == EnumHandlerType.REDUCER
        assert "deterministic" in time_violations[0].message.lower()

    def test_reducer_calling_datetime_now_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Reducer calling datetime.now() detected by AST validator."""
        bad_code = textwrap.dedent("""
            from datetime import datetime

            class OrderReducer:
                def handle(self, event):
                    # Bad: accessing non-deterministic current time
                    current_time = datetime.now()
                    return {"processed_at": current_time}
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        time_violations = [
            v
            for v in violations
            if v.violation_type
            == EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME
        ]
        assert len(time_violations) >= 1

    def test_reducer_calling_datetime_utcnow_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Reducer calling datetime.utcnow() detected by AST validator."""
        bad_code = textwrap.dedent("""
            from datetime import datetime

            class StateReducer:
                def reduce(self, state, action):
                    # Bad: accessing UTC time is still non-deterministic
                    utc_time = datetime.utcnow()
                    return {"last_update": utc_time}
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        time_violations = [
            v
            for v in violations
            if v.violation_type
            == EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME
        ]
        assert len(time_violations) >= 1

    def test_reducer_calling_datetime_datetime_now_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Reducer calling datetime.datetime.now() detected by AST validator."""
        bad_code = textwrap.dedent("""
            import datetime

            class AccountReducer:
                def reduce(self, state, event):
                    # Bad: fully qualified datetime.datetime.now()
                    now = datetime.datetime.now()
                    return {"timestamp": now}
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        time_violations = [
            v
            for v in violations
            if v.violation_type
            == EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME
        ]
        assert len(time_violations) >= 1

    def test_non_reducer_can_access_system_time(self, tmp_path: Path) -> None:
        """Effect handlers are allowed to access system time."""
        valid_code = textwrap.dedent("""
            import time
            from datetime import datetime

            class OrderEffectHandler:
                def handle(self, command):
                    # OK: effect handlers can access system time
                    timestamp = time.time()
                    now = datetime.now()
                    return {"timestamp": timestamp, "now": now}
        """)

        file_path = _write_test_file(tmp_path, valid_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        # No system time violations for effect handlers
        time_violations = [
            v
            for v in violations
            if v.violation_type
            == EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME
        ]
        assert len(time_violations) == 0


class TestHandlerDirectPublishRejected:
    """Test case 5: Any handler directly publishing must be rejected."""

    def test_handler_calling_publish_rejected_by_ast(self, tmp_path: Path) -> None:
        """Handler calling .publish() detected by AST validator."""
        bad_code = textwrap.dedent("""
            class OrderEffectHandler:
                def __init__(self, event_bus):
                    self.event_bus = event_bus

                def handle(self, command):
                    # Bad: directly publishing bypasses event bus abstraction
                    self.event_bus.publish({"type": "OrderCreated"})
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        # Assert HANDLER_DIRECT_PUBLISH violation found
        publish_violations = [
            v
            for v in violations
            if v.violation_type == EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH
        ]
        assert len(publish_violations) >= 1
        assert ".publish()" in publish_violations[0].message

    def test_handler_calling_send_event_rejected_by_ast(
        self, tmp_path: Path
    ) -> None:
        """Handler calling .send_event() detected by AST validator."""
        bad_code = textwrap.dedent("""
            class PaymentReducerHandler:
                def handle(self, event):
                    # Bad: send_event is also direct publishing
                    self.bus.send_event(event)
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        publish_violations = [
            v
            for v in violations
            if v.violation_type == EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH
        ]
        assert len(publish_violations) >= 1
        assert ".send_event()" in publish_violations[0].message

    def test_handler_calling_emit_rejected_by_ast(self, tmp_path: Path) -> None:
        """Handler calling .emit() detected by AST validator."""
        bad_code = textwrap.dedent("""
            class NotificationOrchestrator:
                def orchestrate(self, command):
                    # Bad: emit is also direct publishing
                    self.emitter.emit("notification.sent", {"user": "123"})
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        publish_violations = [
            v
            for v in violations
            if v.violation_type == EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH
        ]
        assert len(publish_violations) >= 1
        assert ".emit()" in publish_violations[0].message

    def test_all_handler_types_forbidden_from_direct_publish(
        self, tmp_path: Path
    ) -> None:
        """All handler types (Effect, Compute, Reducer, Orchestrator) are forbidden."""
        handler_templates = [
            ("OrderEffectHandler", "Effect"),
            ("OrderComputeHandler", "Compute"),
            ("OrderReducerHandler", "Reducer"),
            ("OrderOrchestratorHandler", "Orchestrator"),
        ]

        for idx, (class_name, handler_type_name) in enumerate(handler_templates):
            bad_code = textwrap.dedent(f"""
                class {class_name}:
                    def handle(self, data):
                        self.bus.publish({{"type": "test"}})
            """)

            # Use unique file names for each iteration
            file_path = tmp_path / f"test_handler_{idx}.py"
            file_path.write_text(bad_code)

            validator = ExecutionShapeValidator()
            violations = validator.validate_file(file_path)

            publish_violations = [
                v
                for v in violations
                if v.violation_type == EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH
            ]
            assert len(publish_violations) >= 1, (
                f"{handler_type_name} handler should have direct publish violation"
            )

    def test_dispatch_method_also_rejected(self, tmp_path: Path) -> None:
        """Handler calling .dispatch() is also detected as direct publish."""
        bad_code = textwrap.dedent("""
            class WorkflowOrchestratorHandler:
                def handle(self, command):
                    # Bad: dispatch is also direct publishing
                    self.dispatcher.dispatch(command)
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        publish_violations = [
            v
            for v in violations
            if v.violation_type == EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH
        ]
        assert len(publish_violations) >= 1
        assert ".dispatch()" in publish_violations[0].message


class TestViolationFormatting:
    """Test that violations can be properly formatted for CI output."""

    def test_ast_violation_format_for_ci(self, tmp_path: Path) -> None:
        """AST violations can be formatted for CI output."""
        bad_code = textwrap.dedent("""
            class OrderReducerHandler:
                def handle(self, event) -> OrderCreatedEvent:
                    return OrderCreatedEvent()
        """)

        file_path = _write_test_file(tmp_path, bad_code)
        validator = ExecutionShapeValidator()
        violations = validator.validate_file(file_path)

        assert len(violations) >= 1
        ci_output = violations[0].format_for_ci()
        assert "::error" in ci_output
        assert "reducer_returns_events" in ci_output

    def test_runtime_violation_format_for_ci(self) -> None:
        """Runtime violations can be formatted for CI output."""

        class OrderCreatedEvent:
            category = EnumMessageCategory.EVENT

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            handler_type=EnumHandlerType.REDUCER,
            output=OrderCreatedEvent(),
            output_category=EnumMessageCategory.EVENT,
            file_path="test_handler.py",
            line_number=42,
        )

        assert violation is not None
        ci_output = violation.format_for_ci()
        assert "::error" in ci_output
        assert "test_handler.py" in ci_output
        assert "42" in ci_output


class TestValidHandlers:
    """Verify that valid handlers pass validation (sanity checks)."""

    def test_reducer_returning_projection_is_valid(self) -> None:
        """Reducer returning Projection is valid."""

        class OrderSummaryProjection:
            category = EnumMessageCategory.PROJECTION

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            handler_type=EnumHandlerType.REDUCER,
            output=OrderSummaryProjection(),
            output_category=EnumMessageCategory.PROJECTION,
        )

        assert violation is None

    def test_effect_returning_event_is_valid(self) -> None:
        """Effect returning Event is valid."""

        class OrderCreatedEvent:
            category = EnumMessageCategory.EVENT

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            handler_type=EnumHandlerType.EFFECT,
            output=OrderCreatedEvent(),
            output_category=EnumMessageCategory.EVENT,
        )

        assert violation is None

    def test_orchestrator_returning_command_is_valid(self) -> None:
        """Orchestrator returning Command is valid."""

        class ProcessOrderCommand:
            category = EnumMessageCategory.COMMAND

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            handler_type=EnumHandlerType.ORCHESTRATOR,
            output=ProcessOrderCommand(),
            output_category=EnumMessageCategory.COMMAND,
        )

        assert violation is None

    def test_compute_can_return_any_type(self) -> None:
        """Compute handler can return any message type."""
        validator = RuntimeShapeValidator()

        # Test message class with configurable category attribute.
        # Using a class factory pattern to avoid unconventional class
        # redefinition inside the loop while maintaining test isolation.
        def make_test_message(cat: EnumMessageCategory) -> object:
            """Create a test message instance with the given category."""

            class TestMessage:
                category = cat

            return TestMessage()

        for category in EnumMessageCategory:
            violation = validator.validate_handler_output(
                handler_type=EnumHandlerType.COMPUTE,
                output=make_test_message(category),
                output_category=category,
            )

            assert violation is None, f"Compute should allow {category.value}"


class TestAllowedReturnTypesValidation:
    """Test that allowed_return_types field is used in validation logic.

    These tests verify the is_return_type_allowed() method properly uses
    both allowed_return_types and forbidden_return_types fields.
    """

    def test_allowed_types_enforces_strict_allow_list(self) -> None:
        """When allowed_return_types is specified, only those types are allowed."""
        from omnibase_infra.models.validation.model_execution_shape_rule import (
            ModelExecutionShapeRule,
        )

        # Create a rule that only allows PROJECTION (like REDUCER)
        rule = ModelExecutionShapeRule(
            handler_type=EnumHandlerType.REDUCER,
            allowed_return_types=[EnumMessageCategory.PROJECTION],
            forbidden_return_types=[EnumMessageCategory.EVENT],
            can_publish_directly=False,
            can_access_system_time=False,
        )

        # PROJECTION is allowed (in allowed list)
        assert rule.is_return_type_allowed(EnumMessageCategory.PROJECTION) is True

        # EVENT is forbidden (in forbidden list)
        assert rule.is_return_type_allowed(EnumMessageCategory.EVENT) is False

        # COMMAND is not allowed (not in allowed list)
        assert rule.is_return_type_allowed(EnumMessageCategory.COMMAND) is False

        # INTENT is not allowed (not in allowed list)
        assert rule.is_return_type_allowed(EnumMessageCategory.INTENT) is False

    def test_empty_allowed_list_permits_non_forbidden(self) -> None:
        """When allowed_return_types is empty, all non-forbidden types are allowed."""
        from omnibase_infra.models.validation.model_execution_shape_rule import (
            ModelExecutionShapeRule,
        )

        # Create a rule with empty allowed list but one forbidden type
        rule = ModelExecutionShapeRule(
            handler_type=EnumHandlerType.EFFECT,
            allowed_return_types=[],  # Empty = permissive mode
            forbidden_return_types=[EnumMessageCategory.PROJECTION],
            can_publish_directly=False,
            can_access_system_time=True,
        )

        # PROJECTION is forbidden
        assert rule.is_return_type_allowed(EnumMessageCategory.PROJECTION) is False

        # All others are allowed (empty allowed list = permissive)
        assert rule.is_return_type_allowed(EnumMessageCategory.EVENT) is True
        assert rule.is_return_type_allowed(EnumMessageCategory.COMMAND) is True
        assert rule.is_return_type_allowed(EnumMessageCategory.INTENT) is True

    def test_forbidden_takes_precedence_over_allowed(self) -> None:
        """If a type is in both allowed and forbidden, forbidden wins."""
        from omnibase_infra.models.validation.model_execution_shape_rule import (
            ModelExecutionShapeRule,
        )

        # Create a rule where EVENT is in both lists (edge case)
        rule = ModelExecutionShapeRule(
            handler_type=EnumHandlerType.REDUCER,
            allowed_return_types=[
                EnumMessageCategory.PROJECTION,
                EnumMessageCategory.EVENT,  # Also in forbidden
            ],
            forbidden_return_types=[EnumMessageCategory.EVENT],
            can_publish_directly=False,
            can_access_system_time=False,
        )

        # EVENT should be forbidden (forbidden takes precedence)
        assert rule.is_return_type_allowed(EnumMessageCategory.EVENT) is False

        # PROJECTION should be allowed
        assert rule.is_return_type_allowed(EnumMessageCategory.PROJECTION) is True

    def test_canonical_rules_use_allowed_return_types(self) -> None:
        """Verify canonical execution shape rules properly use allowed_return_types."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        # EFFECT: allowed = [EVENT, COMMAND], forbidden = [PROJECTION]
        effect_rule = EXECUTION_SHAPE_RULES[EnumHandlerType.EFFECT]
        assert effect_rule.is_return_type_allowed(EnumMessageCategory.EVENT) is True
        assert effect_rule.is_return_type_allowed(EnumMessageCategory.COMMAND) is True
        assert effect_rule.is_return_type_allowed(EnumMessageCategory.PROJECTION) is False
        # INTENT is not in allowed list, so should be False
        assert effect_rule.is_return_type_allowed(EnumMessageCategory.INTENT) is False

        # REDUCER: allowed = [PROJECTION], forbidden = [EVENT]
        reducer_rule = EXECUTION_SHAPE_RULES[EnumHandlerType.REDUCER]
        assert reducer_rule.is_return_type_allowed(EnumMessageCategory.PROJECTION) is True
        assert reducer_rule.is_return_type_allowed(EnumMessageCategory.EVENT) is False
        # COMMAND is not in allowed list
        assert reducer_rule.is_return_type_allowed(EnumMessageCategory.COMMAND) is False
        # INTENT is not in allowed list
        assert reducer_rule.is_return_type_allowed(EnumMessageCategory.INTENT) is False

        # ORCHESTRATOR: allowed = [COMMAND, EVENT], forbidden = [INTENT, PROJECTION]
        orch_rule = EXECUTION_SHAPE_RULES[EnumHandlerType.ORCHESTRATOR]
        assert orch_rule.is_return_type_allowed(EnumMessageCategory.COMMAND) is True
        assert orch_rule.is_return_type_allowed(EnumMessageCategory.EVENT) is True
        assert orch_rule.is_return_type_allowed(EnumMessageCategory.INTENT) is False
        assert orch_rule.is_return_type_allowed(EnumMessageCategory.PROJECTION) is False

        # COMPUTE: allowed = [all 4 categories], forbidden = []
        compute_rule = EXECUTION_SHAPE_RULES[EnumHandlerType.COMPUTE]
        for category in EnumMessageCategory:
            assert compute_rule.is_return_type_allowed(category) is True, (
                f"COMPUTE should allow {category.value}"
            )
