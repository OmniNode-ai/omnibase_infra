# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""RED tests for ARCH-001: No Direct Handler Dispatch.

These tests should FAIL initially (RED phase of TDD).
They verify that direct handler dispatch is detected as a violation.

Rule ARCH-001:
    Handlers must not be dispatched directly. All handler invocations must
    go through the runtime dispatcher which provides:
    - Event tracking and correlation
    - Circuit breaking and resilience
    - Metrics and observability
    - Idempotency guarantees

Related:
    - Ticket: OMN-1099 (Architecture Validator)
    - Implementation: validator_no_direct_dispatch.py

Expected Test Outcomes (RED Phase):
    - test_detects_direct_handler_call: FAIL (stub returns valid=True)
    - test_detects_handler_instantiation_and_call: FAIL (stub returns valid=True)
    - test_allows_runtime_dispatch: PASS (stub returns valid=True)
    - test_allows_handler_in_test_files: PASS (stub returns valid=True)
    - test_detects_handler_handle_method_call: FAIL (stub returns valid=True)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.nodes.architecture_validator.validators import (
    validate_no_direct_dispatch,
)


class TestNoDirectHandlerDispatch:
    """Tests for ARCH-001: No Direct Handler Dispatch rule."""

    def test_detects_direct_handler_call(self, tmp_path: Path) -> None:
        """Direct handler.handle() calls should raise violation.

        This test verifies that when code directly calls a handler's
        handle() method, it is detected as an ARCH-001 violation.

        Expected: FAIL in RED phase (stub always returns valid=True)

        """
        # Create a file with forbidden pattern
        bad_code = """
class SomeService:
    def process(self):
        handler = HandlerSomething()
        handler.handle(event)  # VIOLATION: direct dispatch
"""
        test_file = tmp_path / "bad_service.py"
        test_file.write_text(bad_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        # These assertions should FAIL in RED phase
        assert not result.valid, "Expected invalid result for direct handler dispatch"
        assert len(result.violations) == 1, "Expected exactly one violation"
        assert result.violations[0].rule_id == "ARCH-001"
        assert "direct" in result.violations[0].message.lower()

    def test_detects_handler_instantiation_and_call(self, tmp_path: Path) -> None:
        """Handler instantiation followed by call should raise violation.

        This test verifies that the pattern of instantiating a handler
        and then calling its handle() method is detected.

        Expected: FAIL in RED phase (stub always returns valid=True)

        """
        bad_code = """
def process_event(event):
    h = MyHandler(container)
    return h.handle(event)  # VIOLATION
"""
        test_file = tmp_path / "processor.py"
        test_file.write_text(bad_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        # These assertions should FAIL in RED phase
        assert not result.valid, (
            "Expected invalid result for handler instantiation + call"
        )
        assert result.violations[0].rule_id == "ARCH-001"

    def test_detects_handler_handle_method_call(self, tmp_path: Path) -> None:
        """Any .handle() call on handler-like objects should raise violation.

        This test verifies that the validator detects handle() method calls
        even when the handler variable has a generic name.

        Expected: FAIL in RED phase (stub always returns valid=True)

        """
        bad_code = """
class EventProcessor:
    def __init__(self, handler):
        self._handler = handler

    def process(self, event):
        # VIOLATION: Direct dispatch through stored handler
        return self._handler.handle(event)
"""
        test_file = tmp_path / "event_processor.py"
        test_file.write_text(bad_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        # These assertions should FAIL in RED phase
        assert not result.valid, "Expected invalid result for handler attribute call"
        assert len(result.violations) >= 1

    def test_allows_runtime_dispatch(self, tmp_path: Path) -> None:
        """Dispatch through runtime should be allowed.

        This test verifies that proper runtime dispatch patterns
        are not flagged as violations.

        Expected: PASS (stub returns valid=True, which is correct here)

        """
        good_code = """
class Orchestrator:
    def process(self, event):
        # Correct: dispatch through runtime
        return self.runtime.dispatch(event)
"""
        test_file = tmp_path / "good_orchestrator.py"
        test_file.write_text(good_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        assert result.valid, "Runtime dispatch should be allowed"
        assert len(result.violations) == 0

    def test_allows_handler_in_test_files(self, tmp_path: Path) -> None:
        """Direct handler calls in test files should be allowed.

        Test files need to test handlers directly, so they are exempt
        from the ARCH-001 rule.

        Expected: PASS (stub returns valid=True, which is correct here)

        """
        test_code = """
def test_handler_behavior():
    handler = MyHandler(mock_container)
    result = handler.handle(test_event)  # OK in tests
    assert result.success
"""
        # Note: file starts with "test_" to indicate it's a test file
        test_file = tmp_path / "test_handler.py"
        test_file.write_text(test_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        assert result.valid, "Test files should be exempt from ARCH-001"

    def test_detects_multiple_violations_in_single_file(self, tmp_path: Path) -> None:
        """Multiple direct dispatch calls should all be reported.

        This test verifies that when a file contains multiple violations,
        all of them are detected and reported.

        Expected: FAIL in RED phase (stub always returns valid=True)

        """
        bad_code = """
class BadOrchestrator:
    def process_first(self, event):
        handler1 = HandlerOne(self.container)
        return handler1.handle(event)  # VIOLATION 1

    def process_second(self, event):
        handler2 = HandlerTwo(self.container)
        return handler2.handle(event)  # VIOLATION 2

    def process_third(self, event):
        self.handler.handle(event)  # VIOLATION 3
"""
        test_file = tmp_path / "bad_orchestrator.py"
        test_file.write_text(bad_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        # These assertions should FAIL in RED phase
        assert not result.valid, "Expected invalid result for multiple violations"
        assert len(result.violations) >= 3, "Expected at least 3 violations"

    def test_does_not_flag_non_handler_handle_methods(self, tmp_path: Path) -> None:
        """Non-handler classes with handle() methods should not be flagged.

        Some classes legitimately have handle() methods that are not
        handler dispatch calls (e.g., file handles, stream handles).

        Expected: PASS (this is a refinement test for the GREEN phase)

        """
        good_code = """
class FileProcessor:
    def process(self, file_handle):
        # Not a violation - this is a file handle, not a handler
        data = file_handle.read()
        return data
"""
        test_file = tmp_path / "file_processor.py"
        test_file.write_text(good_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        # This should pass - no handle() call present
        assert result.valid, "Non-handler handle methods should not be flagged"

    def test_reports_correct_line_numbers(self, tmp_path: Path) -> None:
        """Violations should include accurate line numbers.

        Line numbers are essential for developers to locate and fix
        violations quickly.

        Expected: FAIL in RED phase (stub returns no violations)

        """
        bad_code = """# Line 1
# Line 2
class Service:  # Line 3
    def process(self):  # Line 4
        handler = MyHandler()  # Line 5
        handler.handle(event)  # Line 6 - VIOLATION HERE
"""
        test_file = tmp_path / "service.py"
        test_file.write_text(bad_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        # These assertions should FAIL in RED phase
        assert not result.valid
        assert len(result.violations) >= 1
        # The violation should be on line 6 (location format is "path:line")
        assert ":6" in result.violations[0].location


class TestNoDirectDispatchEdgeCases:
    """Edge case tests for ARCH-001 validator."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty files should pass validation.

        An empty file has no violations.

        Expected: PASS

        """
        test_file = tmp_path / "empty.py"
        test_file.write_text("", encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        assert result.valid

    def test_file_with_only_comments(self, tmp_path: Path) -> None:
        """Files with only comments should pass validation.

        Expected: PASS

        """
        good_code = '''# This file contains only comments
# handler.handle(event) - this is a comment, not code
"""
handler.handle(event)  # This is a docstring, not code
"""
'''
        test_file = tmp_path / "comments_only.py"
        test_file.write_text(good_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        assert result.valid

    def test_syntax_error_file(self, tmp_path: Path) -> None:
        """Files with syntax errors should be handled gracefully.

        The validator should either skip the file or report a clear error,
        not crash.

        Expected: Implementation-dependent (stub returns valid=True)

        """
        bad_code = """
class Broken
    def missing_colon(self)
        pass
"""
        test_file = tmp_path / "syntax_error.py"
        test_file.write_text(bad_code, encoding="utf-8")

        # Should not raise an exception
        result = validate_no_direct_dispatch(str(test_file))

        # Either valid with a warning, or invalid with syntax error
        # The stub returns valid=True which is acceptable for now
        assert result is not None


class TestNoDirectDispatchFilePatterns:
    """Tests for file pattern handling in ARCH-001 validator."""

    def test_conftest_files_are_exempt(self, tmp_path: Path) -> None:
        """conftest.py files should be exempt (test infrastructure).

        Pytest conftest files are test infrastructure and should be
        allowed to call handlers directly.

        Expected: PASS (stub returns valid=True)

        """
        test_code = """
import pytest

@pytest.fixture
def handler_fixture():
    handler = MyHandler(mock_container)
    result = handler.handle(setup_event)
    return result
"""
        test_file = tmp_path / "conftest.py"
        test_file.write_text(test_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        assert result.valid, "conftest.py should be exempt"

    def test_integration_test_files_are_exempt(self, tmp_path: Path) -> None:
        """Integration test files should be exempt.

        Files in integration test directories need to test the full
        handler behavior.

        Expected: PASS (stub returns valid=True)

        """
        test_code = """
def test_integration():
    handler = RealHandler(real_container)
    result = handler.handle(real_event)
    assert result.success
"""
        # Create in an integration tests directory
        integration_dir = tmp_path / "tests" / "integration"
        integration_dir.mkdir(parents=True)
        test_file = integration_dir / "test_handlers.py"
        test_file.write_text(test_code, encoding="utf-8")

        result = validate_no_direct_dispatch(str(test_file))

        assert result.valid, "Integration test files should be exempt"
