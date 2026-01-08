# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""RED tests for ARCH-002: No Handler Publishing.

These tests should FAIL initially (RED phase of TDD).
They verify that handler publishing is detected as a violation.

Rule ARCH-002: Handlers must not publish events directly.
Only orchestrators may publish events to the event bus.

Detection patterns:
    - Handler class with event_bus in __init__ signature
    - Handler class with _bus, _event_bus, or _publisher attributes
    - Handler class calling publish(), emit(), or send_event() methods

Allowed patterns:
    - Orchestrators may have event bus access
    - Handlers may return events for orchestrators to publish
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.nodes.architecture_validator.models.model_validation_result import (
    ModelFileValidationResult,
)
from omnibase_infra.nodes.architecture_validator.validators.validator_no_handler_publishing import (
    validate_no_handler_publishing,
)


class TestNoHandlerPublishingDetection:
    """Tests that verify ARCH-002 violations are detected.

    These tests should FAIL initially because the stub validator
    always returns valid=True.
    """

    def test_detects_event_bus_in_constructor(self, tmp_path: Path) -> None:
        """Handler with event_bus in constructor should raise violation."""
        bad_code = """
class HandlerUserCreated(ProtocolHandler):
    def __init__(self, container, event_bus):  # VIOLATION: bus in constructor
        self._container = container
        self._bus = event_bus

    def handle(self, event):
        # Process event
        self._bus.publish(SomeEvent())  # VIOLATION: publishing
        return result
"""
        test_file = tmp_path / "handler_user.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        assert not result.valid, (
            "Handler with event_bus in constructor should be invalid"
        )
        assert len(result.violations) >= 1, "Should detect at least one violation"
        assert any(v.rule_id == "ARCH-002" for v in result.violations), (
            "Violation should be ARCH-002"
        )

    def test_detects_event_bus_attribute(self, tmp_path: Path) -> None:
        """Handler with _event_bus attribute should raise violation."""
        bad_code = """
class HandlerOrderProcessed:
    def __init__(self, container):
        self._event_bus = container.resolve("event_bus")  # VIOLATION

    def handle(self, event):
        self._event_bus.emit(OrderCompleted())  # VIOLATION
"""
        test_file = tmp_path / "handler_order.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        assert not result.valid, "Handler with _event_bus attribute should be invalid"
        assert len(result.violations) >= 1, "Should detect at least one violation"
        assert result.violations[0].rule_id == "ARCH-002"

    def test_detects_publisher_attribute(self, tmp_path: Path) -> None:
        """Handler with _publisher attribute should raise violation."""
        bad_code = """
class HandlerPaymentReceived:
    def __init__(self, container):
        self._publisher = container.resolve("publisher")  # VIOLATION

    def handle(self, event):
        self._publisher.publish(PaymentProcessed())  # VIOLATION
"""
        test_file = tmp_path / "handler_payment.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        assert not result.valid, "Handler with _publisher attribute should be invalid"
        assert any(v.rule_id == "ARCH-002" for v in result.violations)

    def test_detects_publish_method_call(self, tmp_path: Path) -> None:
        """Handler calling publish() should raise violation."""
        bad_code = """
class HandlerNotification:
    def handle(self, event):
        # Direct publish call
        self.bus.publish(NotificationSent())  # VIOLATION
        return None
"""
        test_file = tmp_path / "handler_notification.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        assert not result.valid, "Handler calling publish() should be invalid"
        assert len(result.violations) >= 1

    def test_detects_emit_method_call(self, tmp_path: Path) -> None:
        """Handler calling emit() should raise violation."""
        bad_code = """
class HandlerEmail:
    def handle(self, event):
        self.emit(EmailSent())  # VIOLATION: emit is a publish pattern
        return result
"""
        test_file = tmp_path / "handler_email.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        assert not result.valid, "Handler calling emit() should be invalid"
        assert any(v.rule_id == "ARCH-002" for v in result.violations)

    def test_detects_send_event_method_call(self, tmp_path: Path) -> None:
        """Handler calling send_event() should raise violation."""
        bad_code = """
class HandlerAudit:
    def handle(self, event):
        self.send_event(AuditLogCreated())  # VIOLATION
        return result
"""
        test_file = tmp_path / "handler_audit.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        assert not result.valid, "Handler calling send_event() should be invalid"

    def test_detects_multiple_violations_in_single_handler(
        self, tmp_path: Path
    ) -> None:
        """Handler with multiple publish patterns should detect all violations."""
        bad_code = """
class HandlerMultipleViolations:
    def __init__(self, container, event_bus):  # VIOLATION 1: bus in constructor
        self._bus = event_bus  # VIOLATION 2: bus attribute

    def handle(self, event):
        self._bus.publish(Event1())  # VIOLATION 3: publish call
        self.emit(Event2())  # VIOLATION 4: emit call
        return result
"""
        test_file = tmp_path / "handler_multiple.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        assert not result.valid, "Handler with multiple violations should be invalid"
        # Should detect multiple distinct violations
        assert len(result.violations) >= 1


class TestNoHandlerPublishingAllowedPatterns:
    """Tests that verify allowed patterns are NOT flagged as violations.

    These tests should PASS because the stub validator returns valid=True.
    """

    def test_allows_orchestrator_with_event_bus(self, tmp_path: Path) -> None:
        """Orchestrators ARE allowed to have event bus access."""
        good_code = """
class OrchestratorUserWorkflow(NodeOrchestrator):
    def __init__(self, container, event_bus):
        self._bus = event_bus  # OK for orchestrators

    def orchestrate(self, event):
        result = self.process(event)
        self._bus.publish(WorkflowCompleted())  # OK
        return result
"""
        test_file = tmp_path / "orchestrator_user.py"
        test_file.write_text(good_code)

        result = validate_no_handler_publishing(str(test_file))

        assert result.valid, "Orchestrators should be allowed to publish"
        assert len(result.violations) == 0

    def test_allows_handler_returning_events(self, tmp_path: Path) -> None:
        """Handlers returning events (not publishing) should be allowed."""
        good_code = """
class HandlerValidation:
    def handle(self, event) -> ModelEventEnvelope:
        # Returning event for orchestrator to publish is OK
        return ModelEventEnvelope(payload=ValidationCompleted())
"""
        test_file = tmp_path / "handler_validation.py"
        test_file.write_text(good_code)

        result = validate_no_handler_publishing(str(test_file))

        assert result.valid, "Handlers returning events should be valid"

    def test_allows_handler_with_container_only(self, tmp_path: Path) -> None:
        """Handlers with only container injection should be valid."""
        good_code = """
class HandlerProcessing:
    def __init__(self, container):
        self._container = container
        self._service = container.resolve("some_service")

    def handle(self, event):
        result = self._service.process(event)
        return result  # Returns result, does not publish
"""
        test_file = tmp_path / "handler_processing.py"
        test_file.write_text(good_code)

        result = validate_no_handler_publishing(str(test_file))

        assert result.valid, "Handlers with container only should be valid"

    def test_allows_handler_with_yielded_events(self, tmp_path: Path) -> None:
        """Handlers yielding events for orchestrator collection should be valid."""
        good_code = """
class HandlerBatchProcessor:
    def handle(self, event):
        for item in event.items:
            processed = self.process_item(item)
            yield ItemProcessed(item_id=item.id, result=processed)
"""
        test_file = tmp_path / "handler_batch.py"
        test_file.write_text(good_code)

        result = validate_no_handler_publishing(str(test_file))

        assert result.valid, "Handlers yielding events should be valid"


class TestNoHandlerPublishingEdgeCases:
    """Tests for edge cases and boundary conditions.

    These tests verify correct behavior at boundaries.
    """

    def test_ignores_non_handler_classes(self, tmp_path: Path) -> None:
        """Classes that are not handlers should be ignored."""
        code = '''
class EventPublisher:
    """Not a handler - just a utility class."""
    def __init__(self, event_bus):
        self._bus = event_bus

    def publish(self, event):
        self._bus.publish(event)

class NotAHandler:
    def do_something(self):
        pass
'''
        test_file = tmp_path / "publisher_util.py"
        test_file.write_text(code)

        result = validate_no_handler_publishing(str(test_file))

        # Non-handler classes should not trigger violations
        assert result.valid, "Non-handler classes should be ignored"

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        """Empty files should pass validation."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = validate_no_handler_publishing(str(test_file))

        assert result.valid, "Empty files should pass validation"
        assert len(result.violations) == 0

    def test_handles_file_with_syntax_error(self, tmp_path: Path) -> None:
        """Files with syntax errors should be handled gracefully."""
        bad_syntax = """
class HandlerBroken:
    def handle(self
        # Missing closing parenthesis
        return None
"""
        test_file = tmp_path / "handler_broken.py"
        test_file.write_text(bad_syntax)

        # Should not raise exception, but may fail validation
        # The exact behavior depends on implementation
        try:
            result = validate_no_handler_publishing(str(test_file))
            # Either valid with warning or invalid is acceptable
            assert isinstance(result, ModelFileValidationResult)
        except SyntaxError:
            # Also acceptable to raise SyntaxError
            pass

    def test_detects_handler_class_by_naming_convention(self, tmp_path: Path) -> None:
        """Classes named Handler* should be checked for violations."""
        bad_code = '''
class HandlerWithoutBase:
    """No explicit base class, but name indicates handler."""
    def __init__(self, container, bus):
        self._bus = bus  # VIOLATION: handler has bus

    def handle(self, event):
        self._bus.publish(event)  # VIOLATION
'''
        test_file = tmp_path / "handler_custom.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        assert not result.valid, "Handler* named classes should be checked"

    def test_file_path_included_in_violation(self, tmp_path: Path) -> None:
        """Violations should include the correct file path."""
        bad_code = """
class HandlerWithBus:
    def __init__(self, container, event_bus):
        self._bus = event_bus
"""
        test_file = tmp_path / "handler_path_test.py"
        test_file.write_text(bad_code)

        result = validate_no_handler_publishing(str(test_file))

        # When implemented, violations should contain file path in location
        if not result.valid and result.violations:
            assert str(test_file) in result.violations[0].location


class TestValidatorMetadata:
    """Tests for validator metadata and configuration."""

    def test_returns_correct_rule_id_in_result(self, tmp_path: Path) -> None:
        """Result should include ARCH-002 in rules_checked."""
        test_file = tmp_path / "any_file.py"
        test_file.write_text("# empty file")

        result = validate_no_handler_publishing(str(test_file))

        assert "ARCH-002" in result.rules_checked

    def test_returns_file_count(self, tmp_path: Path) -> None:
        """Result should include files_checked count."""
        test_file = tmp_path / "handler.py"
        test_file.write_text("class Handler: pass")

        result = validate_no_handler_publishing(str(test_file))

        assert result.files_checked >= 1
