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
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_archetype import EnumNodeArchetype
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
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
        assert reducer_event_violations[0].node_archetype == EnumNodeArchetype.REDUCER
        assert "event" in reducer_event_violations[0].message.lower()

    def test_reducer_returning_event_call_rejected_by_ast(self, tmp_path: Path) -> None:
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
            node_archetype=EnumNodeArchetype.REDUCER,
            output=OrderCreatedEvent(),
            output_category=EnumMessageCategory.EVENT,
        )

        assert violation is not None
        assert (
            violation.violation_type
            == EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS
        )
        assert violation.node_archetype == EnumNodeArchetype.REDUCER
        assert violation.severity == "error"

    def test_reducer_returning_event_raises_via_decorator(self) -> None:
        """Reducer decorated function raises ExecutionShapeViolationError for EVENT."""

        class OrderCreatedEvent:
            category = EnumMessageCategory.EVENT

        @enforce_execution_shape(EnumNodeArchetype.REDUCER)
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
        assert intent_violations[0].node_archetype == EnumNodeArchetype.ORCHESTRATOR

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
        assert projection_violations[0].node_archetype == EnumNodeArchetype.ORCHESTRATOR

    def test_orchestrator_returning_intent_rejected_by_runtime(self) -> None:
        """Orchestrator returning Intent rejected by runtime validator."""

        class CheckoutIntent:
            category = EnumMessageCategory.INTENT

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            node_archetype=EnumNodeArchetype.ORCHESTRATOR,
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
            category = EnumNodeOutputType.PROJECTION

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            node_archetype=EnumNodeArchetype.ORCHESTRATOR,
            output=OrderProjection(),
            output_category=EnumNodeOutputType.PROJECTION,
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

        @enforce_execution_shape(EnumNodeArchetype.ORCHESTRATOR)
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
            category = EnumNodeOutputType.PROJECTION

        @enforce_execution_shape(EnumNodeArchetype.ORCHESTRATOR)
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
            if v.violation_type
            == EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
        ]
        assert len(projection_violations) >= 1
        assert projection_violations[0].node_archetype == EnumNodeArchetype.EFFECT
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
            if v.violation_type
            == EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
        ]
        assert len(projection_violations) >= 1

    def test_effect_returning_projection_rejected_by_runtime(self) -> None:
        """Effect returning Projection rejected by runtime shape validator."""

        class UserProfileProjection:
            category = EnumNodeOutputType.PROJECTION

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            node_archetype=EnumNodeArchetype.EFFECT,
            output=UserProfileProjection(),
            output_category=EnumNodeOutputType.PROJECTION,
        )

        assert violation is not None
        assert (
            violation.violation_type
            == EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
        )
        assert violation.node_archetype == EnumNodeArchetype.EFFECT
        assert violation.severity == "error"

    def test_effect_returning_projection_raises_via_decorator(self) -> None:
        """Effect decorated function raises ExecutionShapeViolationError for Projection."""

        class UserProfileProjection:
            category = EnumNodeOutputType.PROJECTION

        @enforce_execution_shape(EnumNodeArchetype.EFFECT)
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
        assert time_violations[0].node_archetype == EnumNodeArchetype.REDUCER
        assert "deterministic" in time_violations[0].message.lower()

    def test_reducer_calling_datetime_now_rejected_by_ast(self, tmp_path: Path) -> None:
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

    def test_handler_calling_send_event_rejected_by_ast(self, tmp_path: Path) -> None:
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
                if v.violation_type
                == EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH
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
            node_archetype=EnumNodeArchetype.REDUCER,
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
            category = EnumNodeOutputType.PROJECTION

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            node_archetype=EnumNodeArchetype.REDUCER,
            output=OrderSummaryProjection(),
            output_category=EnumNodeOutputType.PROJECTION,
        )

        assert violation is None

    def test_effect_returning_event_is_valid(self) -> None:
        """Effect returning Event is valid."""

        class OrderCreatedEvent:
            category = EnumMessageCategory.EVENT

        validator = RuntimeShapeValidator()
        violation = validator.validate_handler_output(
            node_archetype=EnumNodeArchetype.EFFECT,
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
            node_archetype=EnumNodeArchetype.ORCHESTRATOR,
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
                node_archetype=EnumNodeArchetype.COMPUTE,
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
            node_archetype=EnumNodeArchetype.REDUCER,
            allowed_return_types=[EnumNodeOutputType.PROJECTION],
            forbidden_return_types=[EnumNodeOutputType.EVENT],
            can_publish_directly=False,
            can_access_system_time=False,
        )

        # PROJECTION is allowed (in allowed list)
        assert rule.is_return_type_allowed(EnumNodeOutputType.PROJECTION) is True

        # EVENT is forbidden (in forbidden list)
        assert rule.is_return_type_allowed(EnumNodeOutputType.EVENT) is False

        # COMMAND is not allowed (not in allowed list)
        assert rule.is_return_type_allowed(EnumNodeOutputType.COMMAND) is False

        # INTENT is not allowed (not in allowed list)
        assert rule.is_return_type_allowed(EnumNodeOutputType.INTENT) is False

    def test_empty_allowed_list_permits_non_forbidden(self) -> None:
        """When allowed_return_types is empty, all non-forbidden types are allowed."""
        from omnibase_infra.models.validation.model_execution_shape_rule import (
            ModelExecutionShapeRule,
        )

        # Create a rule with empty allowed list but one forbidden type
        rule = ModelExecutionShapeRule(
            node_archetype=EnumNodeArchetype.EFFECT,
            allowed_return_types=[],  # Empty = permissive mode
            forbidden_return_types=[EnumNodeOutputType.PROJECTION],
            can_publish_directly=False,
            can_access_system_time=True,
        )

        # PROJECTION is forbidden
        assert rule.is_return_type_allowed(EnumNodeOutputType.PROJECTION) is False

        # All others are allowed (empty allowed list = permissive)
        assert rule.is_return_type_allowed(EnumNodeOutputType.EVENT) is True
        assert rule.is_return_type_allowed(EnumNodeOutputType.COMMAND) is True
        assert rule.is_return_type_allowed(EnumNodeOutputType.INTENT) is True

    def test_forbidden_takes_precedence_over_allowed(self) -> None:
        """If a type is in both allowed and forbidden, forbidden wins."""
        from omnibase_infra.models.validation.model_execution_shape_rule import (
            ModelExecutionShapeRule,
        )

        # Create a rule where EVENT is in both lists (edge case)
        rule = ModelExecutionShapeRule(
            node_archetype=EnumNodeArchetype.REDUCER,
            allowed_return_types=[
                EnumNodeOutputType.PROJECTION,
                EnumNodeOutputType.EVENT,  # Also in forbidden
            ],
            forbidden_return_types=[EnumNodeOutputType.EVENT],
            can_publish_directly=False,
            can_access_system_time=False,
        )

        # EVENT should be forbidden (forbidden takes precedence)
        assert rule.is_return_type_allowed(EnumNodeOutputType.EVENT) is False

        # PROJECTION should be allowed
        assert rule.is_return_type_allowed(EnumNodeOutputType.PROJECTION) is True

    def test_canonical_rules_use_allowed_return_types(self) -> None:
        """Verify canonical execution shape rules properly use allowed_return_types."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        # EFFECT: allowed = [EVENT, COMMAND], forbidden = [PROJECTION]
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]
        assert effect_rule.is_return_type_allowed(EnumNodeOutputType.EVENT) is True
        assert effect_rule.is_return_type_allowed(EnumNodeOutputType.COMMAND) is True
        assert (
            effect_rule.is_return_type_allowed(EnumNodeOutputType.PROJECTION) is False
        )
        # INTENT is not in allowed list, so should be False
        assert effect_rule.is_return_type_allowed(EnumNodeOutputType.INTENT) is False

        # REDUCER: allowed = [PROJECTION], forbidden = [EVENT]
        reducer_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.REDUCER]
        assert (
            reducer_rule.is_return_type_allowed(EnumNodeOutputType.PROJECTION) is True
        )
        assert reducer_rule.is_return_type_allowed(EnumNodeOutputType.EVENT) is False
        # COMMAND is not in allowed list
        assert reducer_rule.is_return_type_allowed(EnumNodeOutputType.COMMAND) is False
        # INTENT is not in allowed list
        assert reducer_rule.is_return_type_allowed(EnumNodeOutputType.INTENT) is False

        # ORCHESTRATOR: allowed = [COMMAND, EVENT], forbidden = [INTENT, PROJECTION]
        orch_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.ORCHESTRATOR]
        assert orch_rule.is_return_type_allowed(EnumNodeOutputType.COMMAND) is True
        assert orch_rule.is_return_type_allowed(EnumNodeOutputType.EVENT) is True
        assert orch_rule.is_return_type_allowed(EnumNodeOutputType.INTENT) is False
        assert orch_rule.is_return_type_allowed(EnumNodeOutputType.PROJECTION) is False

        # COMPUTE: allowed = [all 4 output types], forbidden = []
        compute_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.COMPUTE]
        for output_type in EnumNodeOutputType:
            assert compute_rule.is_return_type_allowed(output_type) is True, (
                f"COMPUTE should allow {output_type.value}"
            )


class TestEnumMappingLogic:
    """Test enum mapping between EnumMessageCategory and EnumNodeOutputType.

    These tests verify that the ExecutionShapeValidator._is_return_type_allowed()
    method correctly handles both enum types and properly maps EnumMessageCategory
    values to EnumNodeOutputType values for validation.

    Context:
        PR #64 introduced EnumNodeOutputType separate from EnumMessageCategory.
        PROJECTION is only in EnumNodeOutputType (not a routable message category).
        The validator must handle both enum types since AST detection may return
        either type depending on how return types are detected in source code.
    """

    def test_message_category_event_maps_correctly(self) -> None:
        """EnumMessageCategory.EVENT maps correctly to EnumNodeOutputType.EVENT."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]

        # EnumMessageCategory.EVENT should work via internal mapping
        result = validator._is_return_type_allowed(
            EnumMessageCategory.EVENT, EnumNodeArchetype.EFFECT, effect_rule
        )
        assert result is True, "EFFECT should allow EVENT via EnumMessageCategory"

    def test_message_category_command_maps_correctly(self) -> None:
        """EnumMessageCategory.COMMAND maps correctly to EnumNodeOutputType.COMMAND."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]

        # EnumMessageCategory.COMMAND should work via internal mapping
        result = validator._is_return_type_allowed(
            EnumMessageCategory.COMMAND, EnumNodeArchetype.EFFECT, effect_rule
        )
        assert result is True, "EFFECT should allow COMMAND via EnumMessageCategory"

    def test_message_category_intent_maps_correctly(self) -> None:
        """EnumMessageCategory.INTENT maps correctly to EnumNodeOutputType.INTENT."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        compute_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.COMPUTE]

        # EnumMessageCategory.INTENT should work via internal mapping
        result = validator._is_return_type_allowed(
            EnumMessageCategory.INTENT, EnumNodeArchetype.COMPUTE, compute_rule
        )
        assert result is True, "COMPUTE should allow INTENT via EnumMessageCategory"

    def test_node_output_type_works_directly(self) -> None:
        """EnumNodeOutputType values work directly without mapping."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]
        reducer_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.REDUCER]

        # EnumNodeOutputType.EVENT should work directly
        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.EVENT, EnumNodeArchetype.EFFECT, effect_rule
            )
            is True
        )

        # EnumNodeOutputType.PROJECTION should work directly for REDUCER
        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.PROJECTION, EnumNodeArchetype.REDUCER, reducer_rule
            )
            is True
        )

    def test_mixed_enum_types_in_validation(self) -> None:
        """Both enum types produce consistent validation results."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]

        # EnumMessageCategory.EVENT and EnumNodeOutputType.EVENT should behave identically
        result_message_cat = validator._is_return_type_allowed(
            EnumMessageCategory.EVENT, EnumNodeArchetype.EFFECT, effect_rule
        )
        result_node_output = validator._is_return_type_allowed(
            EnumNodeOutputType.EVENT, EnumNodeArchetype.EFFECT, effect_rule
        )
        assert result_message_cat == result_node_output, (
            "Both enum types should produce consistent results for EVENT"
        )


class TestProjectionOnlyAllowedForReducer:
    """Test that PROJECTION is a node output type only valid for REDUCERs.

    PROJECTION represents state consolidation output and is NOT a message
    routing category. It exists in EnumNodeOutputType but not EnumMessageCategory,
    and can only be produced by REDUCER handlers.
    """

    def test_projection_allowed_for_reducer(self) -> None:
        """PROJECTION is allowed as output for REDUCER handlers."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        reducer_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.REDUCER]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.PROJECTION, EnumNodeArchetype.REDUCER, reducer_rule
            )
            is True
        ), "REDUCER should be allowed to produce PROJECTION"

    def test_projection_not_allowed_for_effect(self) -> None:
        """PROJECTION is NOT allowed as output for EFFECT handlers."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.PROJECTION, EnumNodeArchetype.EFFECT, effect_rule
            )
            is False
        ), "EFFECT should NOT be allowed to produce PROJECTION"

    def test_projection_not_allowed_for_orchestrator(self) -> None:
        """PROJECTION is NOT allowed as output for ORCHESTRATOR handlers."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        orchestrator_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.ORCHESTRATOR]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.PROJECTION,
                EnumNodeArchetype.ORCHESTRATOR,
                orchestrator_rule,
            )
            is False
        ), "ORCHESTRATOR should NOT be allowed to produce PROJECTION"

    def test_projection_allowed_for_compute(self) -> None:
        """PROJECTION is allowed as output for COMPUTE handlers (most permissive)."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        compute_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.COMPUTE]

        # COMPUTE is the most permissive handler type - allows all output types
        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.PROJECTION, EnumNodeArchetype.COMPUTE, compute_rule
            )
            is True
        ), "COMPUTE should be allowed to produce PROJECTION (most permissive)"


class TestHandlerTypeOutputRestrictions:
    """Comprehensive tests for handler-specific output type restrictions.

    Each handler type has specific constraints on what output types it can produce:
    - EFFECT: Can return EVENT, COMMAND but NOT PROJECTION or INTENT
    - REDUCER: Can only return PROJECTION, NOT EVENT, COMMAND, or INTENT
    - ORCHESTRATOR: Can return EVENT, COMMAND but NOT INTENT or PROJECTION
    - COMPUTE: Can return any type (most permissive)
    """

    def test_effect_can_return_event(self) -> None:
        """EFFECT handlers can return EVENT type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.EVENT, EnumNodeArchetype.EFFECT, effect_rule
            )
            is True
        )

    def test_effect_can_return_command(self) -> None:
        """EFFECT handlers can return COMMAND type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.COMMAND, EnumNodeArchetype.EFFECT, effect_rule
            )
            is True
        )

    def test_effect_cannot_return_intent(self) -> None:
        """EFFECT handlers cannot return INTENT type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]

        # INTENT is not in EFFECT's allowed_return_types
        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.INTENT, EnumNodeArchetype.EFFECT, effect_rule
            )
            is False
        )

    def test_effect_cannot_return_projection(self) -> None:
        """EFFECT handlers cannot return PROJECTION type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        effect_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.EFFECT]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.PROJECTION, EnumNodeArchetype.EFFECT, effect_rule
            )
            is False
        )

    def test_reducer_can_return_projection(self) -> None:
        """REDUCER handlers can only return PROJECTION type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        reducer_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.REDUCER]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.PROJECTION, EnumNodeArchetype.REDUCER, reducer_rule
            )
            is True
        )

    def test_reducer_cannot_return_event(self) -> None:
        """REDUCER handlers cannot return EVENT type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        reducer_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.REDUCER]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.EVENT, EnumNodeArchetype.REDUCER, reducer_rule
            )
            is False
        )

    def test_reducer_cannot_return_command(self) -> None:
        """REDUCER handlers cannot return COMMAND type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        reducer_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.REDUCER]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.COMMAND, EnumNodeArchetype.REDUCER, reducer_rule
            )
            is False
        )

    def test_reducer_cannot_return_intent(self) -> None:
        """REDUCER handlers cannot return INTENT type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        reducer_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.REDUCER]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.INTENT, EnumNodeArchetype.REDUCER, reducer_rule
            )
            is False
        )

    def test_orchestrator_can_return_event(self) -> None:
        """ORCHESTRATOR handlers can return EVENT type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        orch_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.ORCHESTRATOR]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.EVENT, EnumNodeArchetype.ORCHESTRATOR, orch_rule
            )
            is True
        )

    def test_orchestrator_can_return_command(self) -> None:
        """ORCHESTRATOR handlers can return COMMAND type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        orch_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.ORCHESTRATOR]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.COMMAND, EnumNodeArchetype.ORCHESTRATOR, orch_rule
            )
            is True
        )

    def test_orchestrator_cannot_return_intent(self) -> None:
        """ORCHESTRATOR handlers cannot return INTENT type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        orch_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.ORCHESTRATOR]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.INTENT, EnumNodeArchetype.ORCHESTRATOR, orch_rule
            )
            is False
        )

    def test_orchestrator_cannot_return_projection(self) -> None:
        """ORCHESTRATOR handlers cannot return PROJECTION type."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        orch_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.ORCHESTRATOR]

        assert (
            validator._is_return_type_allowed(
                EnumNodeOutputType.PROJECTION, EnumNodeArchetype.ORCHESTRATOR, orch_rule
            )
            is False
        )

    def test_compute_can_return_any_type(self) -> None:
        """COMPUTE handlers can return any output type (most permissive)."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        compute_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.COMPUTE]

        # COMPUTE is the most permissive - should allow all output types
        for output_type in EnumNodeOutputType:
            assert (
                validator._is_return_type_allowed(
                    output_type, EnumNodeArchetype.COMPUTE, compute_rule
                )
                is True
            ), f"COMPUTE should allow {output_type.value}"


class TestEnumMappingEdgeCases:
    """Test edge cases and forward compatibility for enum mapping.

    These tests verify that the mapping logic handles edge cases gracefully
    and provides future-proof behavior for potential enum extensions.
    """

    def test_all_message_categories_have_output_type_mapping(self) -> None:
        """Every EnumMessageCategory value has a corresponding EnumNodeOutputType."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        compute_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.COMPUTE]

        # For each message category, the validator should return True or False
        # (not raise an exception) when called with COMPUTE (most permissive)
        for category in EnumMessageCategory:
            result = validator._is_return_type_allowed(
                category, EnumNodeArchetype.COMPUTE, compute_rule
            )
            # COMPUTE allows all mapped types, so all should be True
            assert result is True, (
                f"EnumMessageCategory.{category.name} should map and be allowed for COMPUTE"
            )

    def test_all_node_output_types_handled(self) -> None:
        """Every EnumNodeOutputType value is properly handled."""
        from omnibase_infra.validation.execution_shape_validator import (
            EXECUTION_SHAPE_RULES,
        )

        validator = ExecutionShapeValidator()
        compute_rule = EXECUTION_SHAPE_RULES[EnumNodeArchetype.COMPUTE]

        for output_type in EnumNodeOutputType:
            result = validator._is_return_type_allowed(
                output_type, EnumNodeArchetype.COMPUTE, compute_rule
            )
            assert result is True, (
                f"EnumNodeOutputType.{output_type.name} should be allowed for COMPUTE"
            )

    def test_message_category_to_output_type_value_consistency(self) -> None:
        """EnumMessageCategory and EnumNodeOutputType share consistent string values.

        The mapping relies on both enums having the same string values for
        EVENT, COMMAND, and INTENT. This test verifies that assumption.
        """
        # Shared categories should have identical string values
        assert EnumMessageCategory.EVENT.value == EnumNodeOutputType.EVENT.value
        assert EnumMessageCategory.COMMAND.value == EnumNodeOutputType.COMMAND.value
        assert EnumMessageCategory.INTENT.value == EnumNodeOutputType.INTENT.value

    def test_projection_not_in_message_category(self) -> None:
        """PROJECTION exists only in EnumNodeOutputType, not EnumMessageCategory.

        This is intentional: PROJECTION is a node output type for state
        consolidation, not a message routing category for Kafka topics.
        """
        # EnumMessageCategory should not have PROJECTION
        message_category_names = {m.name for m in EnumMessageCategory}
        assert "PROJECTION" not in message_category_names, (
            "PROJECTION should not be a message routing category"
        )

        # EnumNodeOutputType should have PROJECTION
        node_output_names = {o.name for o in EnumNodeOutputType}
        assert "PROJECTION" in node_output_names, (
            "PROJECTION should be a valid node output type"
        )


class TestEnumMappingCompleteness:
    """Test that enum mappings stay in sync with enum definitions.

    These tests ensure that when new values are added to EnumMessageCategory,
    the corresponding mapping in _MESSAGE_CATEGORY_TO_OUTPUT_TYPE is also updated.
    This prevents drift/sync issues between the enum and the mapping.

    Context:
        OMN-974 introduced EnumNodeOutputType separate from EnumMessageCategory.
        The _MESSAGE_CATEGORY_TO_OUTPUT_TYPE mapping bridges these two enums
        for execution shape validation. If someone adds a new EnumMessageCategory
        value but forgets to add it to the mapping, validation will silently
        fail (returning False for unknown categories).

    This test class acts as a guard rail to catch such omissions.
    """

    def test_all_message_categories_exist_in_output_type_mapping(self) -> None:
        """Verify every EnumMessageCategory value has a mapping to EnumNodeOutputType.

        This test will FAIL if someone adds a new value to EnumMessageCategory
        but forgets to add the corresponding entry in _MESSAGE_CATEGORY_TO_OUTPUT_TYPE.

        The mapping is critical for the ExecutionShapeValidator._is_return_type_allowed()
        method which converts EnumMessageCategory values to EnumNodeOutputType for
        validation against execution shape rules.
        """
        from omnibase_infra.validation.execution_shape_validator import (
            _MESSAGE_CATEGORY_TO_OUTPUT_TYPE,
        )

        # Get all EnumMessageCategory values
        all_categories = set(EnumMessageCategory)

        # Get all categories that have mappings
        mapped_categories = set(_MESSAGE_CATEGORY_TO_OUTPUT_TYPE.keys())

        # Check for missing mappings
        missing_mappings = all_categories - mapped_categories

        assert not missing_mappings, (
            f"The following EnumMessageCategory values are missing from "
            f"_MESSAGE_CATEGORY_TO_OUTPUT_TYPE mapping: {missing_mappings}. "
            f"When adding new values to EnumMessageCategory, you MUST also update "
            f"the _MESSAGE_CATEGORY_TO_OUTPUT_TYPE mapping in "
            f"omnibase_infra/validation/execution_shape_validator.py"
        )

    def test_mapping_values_are_valid_node_output_types(self) -> None:
        """Verify all mapping values are valid EnumNodeOutputType members.

        This ensures the mapping doesn't contain typos or invalid output types.
        """
        from omnibase_infra.validation.execution_shape_validator import (
            _MESSAGE_CATEGORY_TO_OUTPUT_TYPE,
        )

        for category, output_type in _MESSAGE_CATEGORY_TO_OUTPUT_TYPE.items():
            assert isinstance(output_type, EnumNodeOutputType), (
                f"Mapping for {category} must be an EnumNodeOutputType, "
                f"got {type(output_type).__name__}: {output_type}"
            )

    def test_mapping_preserves_semantic_equivalence(self) -> None:
        """Verify mapped categories have matching semantic values.

        EVENT, COMMAND, and INTENT should map to their EnumNodeOutputType
        counterparts with identical string values, ensuring semantic consistency.
        """
        from omnibase_infra.validation.execution_shape_validator import (
            _MESSAGE_CATEGORY_TO_OUTPUT_TYPE,
        )

        for category, output_type in _MESSAGE_CATEGORY_TO_OUTPUT_TYPE.items():
            assert category.value == output_type.value, (
                f"EnumMessageCategory.{category.name} (value={category.value!r}) "
                f"should map to EnumNodeOutputType with same value, but maps to "
                f"EnumNodeOutputType.{output_type.name} (value={output_type.value!r})"
            )

    def test_no_extra_mappings_for_nonexistent_categories(self) -> None:
        """Verify mapping doesn't contain stale entries for removed categories.

        If an EnumMessageCategory value is ever removed, this test ensures
        the mapping is also cleaned up.
        """
        from omnibase_infra.validation.execution_shape_validator import (
            _MESSAGE_CATEGORY_TO_OUTPUT_TYPE,
        )

        all_categories = set(EnumMessageCategory)
        mapped_categories = set(_MESSAGE_CATEGORY_TO_OUTPUT_TYPE.keys())

        extra_mappings = mapped_categories - all_categories

        assert not extra_mappings, (
            f"_MESSAGE_CATEGORY_TO_OUTPUT_TYPE contains mappings for categories "
            f"that no longer exist in EnumMessageCategory: {extra_mappings}. "
            f"Remove these stale entries from the mapping."
        )
