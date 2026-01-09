# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Property-based and edge case tests for architecture validator Rule classes.

This module provides comprehensive edge case testing for the AST-based architecture
validators to ensure they handle unusual inputs gracefully without crashing.

Test Coverage:
    1. Property-based tests with hypothesis for random but valid Python code
    2. Edge cases:
       - Empty files
       - Files with only comments/docstrings
       - Files with syntax errors
       - Files with unicode identifiers
       - Files with deeply nested code
       - Large files with many classes
       - Files with unusual but valid Python constructs

Related:
    - PR Review: #124 suggested property-based testing for AST edge cases
    - Ticket: OMN-1099 (Architecture Validator)

Note:
    These tests ensure validators never crash on any input and always return
    valid ModelRuleCheckResult or ModelFileValidationResult objects.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.nodes.architecture_validator.validators import (
    RuleNoDirectDispatch,
    RuleNoHandlerPublishing,
    RuleNoOrchestratorFSM,
)
from omnibase_infra.nodes.architecture_validator.validators.validator_no_direct_dispatch import (
    validate_no_direct_dispatch,
)
from omnibase_infra.nodes.architecture_validator.validators.validator_no_handler_publishing import (
    validate_no_handler_publishing,
)
from omnibase_infra.nodes.architecture_validator.validators.validator_no_orchestrator_fsm import (
    validate_no_orchestrator_fsm,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.architecture_validator.models import (
        ModelRuleCheckResult,
    )
    from omnibase_infra.nodes.architecture_validator.models.model_validation_result import (
        ModelFileValidationResult,
    )

# Try to import hypothesis - skip property tests if not available
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rule_no_direct_dispatch() -> RuleNoDirectDispatch:
    """Create a RuleNoDirectDispatch instance."""
    return RuleNoDirectDispatch()


@pytest.fixture
def rule_no_handler_publishing() -> RuleNoHandlerPublishing:
    """Create a RuleNoHandlerPublishing instance."""
    return RuleNoHandlerPublishing()


@pytest.fixture
def rule_no_orchestrator_fsm() -> RuleNoOrchestratorFSM:
    """Create a RuleNoOrchestratorFSM instance."""
    return RuleNoOrchestratorFSM()


@pytest.fixture
def create_temp_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary files.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture.

    Returns:
        Callable that creates temp files with given name and content.
    """

    def _create(filename: str, content: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create


@pytest.fixture
def create_temp_file_bytes(tmp_path: Path) -> Callable[[str, bytes], Path]:
    """Factory fixture for creating temporary files with raw bytes.

    Useful for testing encoding error handling.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture.

    Returns:
        Callable that creates temp files with given name and bytes content.
    """

    def _create(filename: str, content: bytes) -> Path:
        file_path = tmp_path / filename
        file_path.write_bytes(content)
        return file_path

    return _create


# =============================================================================
# Helper Functions
# =============================================================================


def assert_result_is_valid(result: ModelRuleCheckResult) -> None:
    """Assert that a ModelRuleCheckResult is structurally valid.

    Args:
        result: The result to validate.
    """
    assert hasattr(result, "passed")
    assert hasattr(result, "rule_id")
    assert isinstance(result.passed, bool)
    assert isinstance(result.rule_id, str)


def assert_file_result_is_valid(result: ModelFileValidationResult) -> None:
    """Assert that a ModelFileValidationResult is structurally valid.

    Args:
        result: The result to validate.
    """
    assert hasattr(result, "valid")
    assert hasattr(result, "violations")
    assert isinstance(result.valid, bool)
    assert isinstance(result.violations, list)


# =============================================================================
# Edge Case Tests: Empty and Minimal Files
# =============================================================================


class TestEdgeCaseEmptyFiles:
    """Tests for empty and minimal file handling."""

    def test_empty_file_no_direct_dispatch(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Empty Python file should be handled gracefully."""
        file_path = create_temp_file("empty.py", "")
        result = rule_no_direct_dispatch.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_empty_file_no_handler_publishing(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Empty Python file should be handled gracefully."""
        file_path = create_temp_file("empty.py", "")
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_empty_file_no_orchestrator_fsm(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Empty Python file should be handled gracefully."""
        file_path = create_temp_file("empty.py", "")
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_whitespace_only_file(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with only whitespace should be handled gracefully."""
        file_path = create_temp_file("whitespace.py", "   \n\n\t\t\n   ")
        result = rule_no_direct_dispatch.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_newlines_only_file(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with only newlines should be handled gracefully."""
        file_path = create_temp_file("newlines.py", "\n" * 100)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True


class TestEdgeCaseCommentOnlyFiles:
    """Tests for files containing only comments or docstrings."""

    def test_comment_only_file(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with only comments should be handled gracefully."""
        content = """# This is a comment
# Another comment
# Yet another comment
"""
        file_path = create_temp_file("comments.py", content)
        result = rule_no_direct_dispatch.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_docstring_only_file(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with only a module docstring should be handled gracefully."""
        content = '''"""This is a module docstring.

It contains multiple lines of documentation.
"""
'''
        file_path = create_temp_file("docstring.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_multiline_string_file(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with multiline string expressions should be handled gracefully."""
        content = """'''
Triple single quote string
spanning multiple lines
'''

\"\"\"
Triple double quote string
also spanning multiple lines
\"\"\"
"""
        file_path = create_temp_file("strings.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True


# =============================================================================
# Edge Case Tests: Syntax Errors
# =============================================================================


class TestEdgeCaseSyntaxErrors:
    """Tests for files with syntax errors."""

    def test_syntax_error_unclosed_paren(
        self,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with unclosed parenthesis should return warning, not crash."""
        content = """
class Handler:
    def handle(self, event:
        return event
"""
        file_path = create_temp_file("syntax_error.py", content)
        result = validate_no_handler_publishing(str(file_path))
        assert_file_result_is_valid(result)
        # Should be valid (not a rule violation) but may have warning
        assert result.valid is True
        # Should have a warning about syntax error
        if result.violations:
            assert any("syntax" in v.message.lower() for v in result.violations)

    def test_syntax_error_invalid_token(
        self,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with invalid token should return warning, not crash."""
        content = """
class Orchestrator:
    def orchestrate(self):
        return @@@  # Invalid syntax
"""
        file_path = create_temp_file("invalid_token.py", content)
        result = validate_no_orchestrator_fsm(str(file_path))
        assert_file_result_is_valid(result)
        assert result.valid is True

    def test_syntax_error_incomplete_class(
        self,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with incomplete class should return warning, not crash."""
        content = """
class IncompleteHandler:
"""
        file_path = create_temp_file("incomplete.py", content)
        result = validate_no_direct_dispatch(str(file_path))
        assert_file_result_is_valid(result)
        # Incomplete class is actually valid Python (pass is implicit at EOF)
        # but may trigger a warning depending on Python version
        assert result.valid is True

    def test_syntax_error_mixed_tabs_spaces(
        self,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with mixed indentation might cause issues in some Python versions."""
        # This is actually valid Python 3, but good to test
        content = """
class Handler:
    def handle(self, event):
\t    return event  # Mixed tab and spaces
"""
        file_path = create_temp_file("mixed_indent.py", content)
        result = validate_no_handler_publishing(str(file_path))
        assert_file_result_is_valid(result)


# =============================================================================
# Edge Case Tests: Unicode and Special Characters
# =============================================================================


class TestEdgeCaseUnicode:
    """Tests for files with unicode identifiers and content."""

    def test_unicode_class_name(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with unicode class name should be handled gracefully."""
        content = """
class HandlerUnicode_\u00e9v\u00e8nement:
    def handle(self, event):
        return event
"""
        file_path = create_temp_file("unicode_class.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_unicode_method_name(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with unicode method name should be handled gracefully."""
        content = """
class OrchestratorTest:
    def orchestrate_\u4e2d\u6587(self, event):
        return event
"""
        file_path = create_temp_file("unicode_method.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_unicode_variable_name(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with unicode variable name should be handled gracefully."""
        content = """
class Service:
    def process(self):
        \u03b1_handler = "not a real handler"
        \u03b2_result = self.runtime.dispatch(event)
        return \u03b2_result
"""
        file_path = create_temp_file("unicode_var.py", content)
        result = rule_no_direct_dispatch.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_emoji_in_comments(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with emoji in comments should be handled gracefully."""
        content = """
# This is a Handler class 🎉
class Handler:
    def handle(self, event):  # Process event 🚀
        return event  # Return result ✅
"""
        file_path = create_temp_file("emoji.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_emoji_in_strings(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with emoji in strings should be handled gracefully."""
        content = """
class Orchestrator:
    MESSAGE = "Hello 🌍 World 🎊"

    def orchestrate(self, event):
        return f"Processing {event} 🔄"
"""
        file_path = create_temp_file("emoji_strings.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True


class TestEdgeCaseEncodingErrors:
    """Tests for files with encoding issues."""

    def test_invalid_utf8_bytes(
        self,
        create_temp_file_bytes: Callable[[str, bytes], Path],
    ) -> None:
        """File with invalid UTF-8 bytes should return warning, not crash."""
        # Invalid UTF-8 sequence
        content = b"class Handler:\n    def handle(self):\n        return \xff\xfe"
        file_path = create_temp_file_bytes("invalid_utf8.py", content)
        result = validate_no_handler_publishing(str(file_path))
        assert_file_result_is_valid(result)
        assert result.valid is True  # Still valid (graceful handling)

    def test_latin1_encoded_file(
        self,
        create_temp_file_bytes: Callable[[str, bytes], Path],
    ) -> None:
        """File with Latin-1 encoding should be handled gracefully."""
        # Latin-1 encoded content that's not valid UTF-8
        content = "class Handler:\n    name = '\xe9v\xe8nement'\n".encode("latin-1")
        file_path = create_temp_file_bytes("latin1.py", content)
        result = validate_no_direct_dispatch(str(file_path))
        assert_file_result_is_valid(result)
        # May fail to decode, but should not crash


# =============================================================================
# Edge Case Tests: Deeply Nested Code
# =============================================================================


class TestEdgeCaseDeeplyNested:
    """Tests for files with deeply nested code structures."""

    def test_deeply_nested_handler(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Deeply nested handler class should be analyzed correctly."""
        content = """
class Outer:
    class Middle:
        class Inner:
            class VeryDeep:
                class HandlerNested:
                    def __init__(self, container):
                        self._container = container

                    def handle(self, event):
                        return event
"""
        file_path = create_temp_file("nested_handler.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        # Should pass since no bus access
        assert result.passed is True

    def test_deeply_nested_orchestrator(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Deeply nested orchestrator should be analyzed correctly."""
        content = """
class Namespace:
    class Domain:
        class Module:
            class OrchestratorDeep:
                def orchestrate(self, event):
                    return event
"""
        file_path = create_temp_file("nested_orchestrator.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_deeply_nested_function_calls(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Deeply nested function calls should be analyzed correctly."""
        content = """
class Service:
    def process(self, event):
        return (
            self.transform(
                self.validate(
                    self.parse(
                        self.decode(
                            self.receive(
                                self.runtime.dispatch(event)
                            )
                        )
                    )
                )
            )
        )
"""
        file_path = create_temp_file("nested_calls.py", content)
        result = rule_no_direct_dispatch.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_deeply_nested_conditionals(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Deeply nested conditionals should be analyzed correctly."""
        content = """
class Handler:
    def handle(self, event):
        if event.type == "a":
            if event.subtype == "b":
                if event.category == "c":
                    if event.priority == "d":
                        if event.status == "e":
                            return "deep"
        return "shallow"
"""
        file_path = create_temp_file("nested_conditionals.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True


# =============================================================================
# Edge Case Tests: Large Files
# =============================================================================


class TestEdgeCaseLargeFiles:
    """Tests for large files with many classes/methods."""

    def test_many_handler_classes(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with many handler classes should complete in reasonable time."""
        # Generate 50 handler classes
        classes = []
        for i in range(50):
            classes.append(f"""
class Handler{i}:
    def __init__(self, container):
        self._container = container

    def handle(self, event):
        return event.data + {i}
""")
        content = "\n".join(classes)
        file_path = create_temp_file("many_handlers.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_many_orchestrator_classes(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with many orchestrator classes should complete in reasonable time."""
        # Generate 50 orchestrator classes
        classes = []
        for i in range(50):
            classes.append(f"""
class Orchestrator{i}:
    def orchestrate(self, event):
        return self.plan_reactions(event.type + "{i}")
""")
        content = "\n".join(classes)
        file_path = create_temp_file("many_orchestrators.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_class_with_many_methods(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Class with many methods should be analyzed completely."""
        methods = []
        for i in range(100):
            methods.append(f"""
    def method_{i}(self, arg):
        return self.runtime.dispatch(arg)
""")
        content = "class Service:\n" + "\n".join(methods)
        file_path = create_temp_file("many_methods.py", content)
        result = rule_no_direct_dispatch.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_large_file_with_violations(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Large file with a violation should still detect it."""
        # Generate many clean handlers, then one with violation
        classes = []
        for i in range(25):
            classes.append(f"""
class Handler{i}:
    def handle(self, event):
        return event
""")
        # Add violating handler
        classes.append("""
class HandlerBad:
    def __init__(self, container, event_bus):
        self._bus = event_bus

    def handle(self, event):
        self._bus.publish(event)
""")
        # Add more clean handlers
        for i in range(25, 50):
            classes.append(f"""
class Handler{i}:
    def handle(self, event):
        return event
""")
        content = "\n".join(classes)
        file_path = create_temp_file("large_with_violation.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        # Should detect the violation
        assert result.passed is False


# =============================================================================
# Edge Case Tests: Unusual but Valid Python
# =============================================================================


class TestEdgeCaseUnusualPython:
    """Tests for unusual but valid Python constructs."""

    def test_async_handler(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Async handler should be analyzed correctly."""
        content = """
class Handler:
    async def __init__(self, container):
        self._container = container

    async def handle(self, event):
        await asyncio.sleep(0)
        return event
"""
        file_path = create_temp_file("async_handler.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_decorated_methods(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Decorated methods should be analyzed correctly."""
        content = """
class Orchestrator:
    @staticmethod
    def static_helper():
        return "helper"

    @classmethod
    def class_helper(cls):
        return cls.__name__

    @property
    def name(self):
        return "orchestrator"

    @functools.lru_cache(maxsize=128)
    def cached_method(self, key):
        return key * 2

    def orchestrate(self, event):
        return event
"""
        file_path = create_temp_file("decorated.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_metaclass_handler(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Handler with metaclass should be analyzed correctly."""
        content = """
class Meta(type):
    pass

class Handler(metaclass=Meta):
    def handle(self, event):
        return event
"""
        file_path = create_temp_file("metaclass.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_multiple_inheritance(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Class with multiple inheritance should be analyzed correctly."""
        content = """
class Mixin1:
    def helper1(self):
        pass

class Mixin2:
    def helper2(self):
        pass

class OrchestratorMulti(Mixin1, Mixin2):
    def orchestrate(self, event):
        self.helper1()
        self.helper2()
        return event
"""
        file_path = create_temp_file("multi_inherit.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_dataclass_handler(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Dataclass-based handler should be analyzed correctly."""
        content = """
from dataclasses import dataclass

@dataclass
class HandlerConfig:
    timeout: int = 30
    retries: int = 3

class Handler:
    def __init__(self, config: HandlerConfig):
        self._config = config

    def handle(self, event):
        return event
"""
        file_path = create_temp_file("dataclass.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_walrus_operator(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Code with walrus operator should be analyzed correctly."""
        content = """
class Service:
    def process(self, event):
        if (result := self.runtime.dispatch(event)) is not None:
            return result
        return None
"""
        file_path = create_temp_file("walrus.py", content)
        result = rule_no_direct_dispatch.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_match_statement(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Code with match statement (Python 3.10+) should be analyzed correctly."""
        content = """
class Orchestrator:
    def orchestrate(self, event):
        match event.type:
            case "create":
                return self.create_handler(event)
            case "update":
                return self.update_handler(event)
            case "delete":
                return self.delete_handler(event)
            case _:
                return None
"""
        file_path = create_temp_file("match.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_type_hints_complex(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Complex type hints should not interfere with analysis."""
        content = """
from typing import Generic, TypeVar, Protocol, Callable

T = TypeVar("T")
E = TypeVar("E", bound="Event")

class Handler(Generic[T, E]):
    def __init__(
        self,
        container: "Container[T]",
        callback: Callable[[E], T | None],
    ) -> None:
        self._container = container
        self._callback = callback

    def handle(self, event: E) -> T | None:
        return self._callback(event)
"""
        file_path = create_temp_file("complex_types.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True


# =============================================================================
# Property-Based Tests with Hypothesis
# =============================================================================


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBasedValidation:
    """Property-based tests using hypothesis to generate random inputs."""

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_string_no_crash_direct_dispatch(self, text: str) -> None:
        """RuleNoDirectDispatch.check() should never crash on arbitrary strings."""
        rule = RuleNoDirectDispatch()
        result = rule.check(text)
        assert_result_is_valid(result)

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_string_no_crash_handler_publishing(self, text: str) -> None:
        """RuleNoHandlerPublishing.check() should never crash on arbitrary strings."""
        rule = RuleNoHandlerPublishing()
        result = rule.check(text)
        assert_result_is_valid(result)

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_string_no_crash_orchestrator_fsm(self, text: str) -> None:
        """RuleNoOrchestratorFSM.check() should never crash on arbitrary strings."""
        rule = RuleNoOrchestratorFSM()
        result = rule.check(text)
        assert_result_is_valid(result)

    @given(
        st.one_of(
            st.none(),
            st.integers(),
            st.floats(allow_nan=False),
            st.booleans(),
            st.binary(max_size=50),  # Limit binary size to avoid path length issues
            st.lists(st.integers(), max_size=5),  # Limit list size
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_type_no_crash(self, value: object) -> None:
        """All rule classes should handle arbitrary types without crashing.

        Note: We limit collection sizes to avoid OSError from very long string
        representations being used as file paths.
        """
        for rule_class in [
            RuleNoDirectDispatch,
            RuleNoHandlerPublishing,
            RuleNoOrchestratorFSM,
        ]:
            rule = rule_class()
            result = rule.check(value)
            assert_result_is_valid(result)
            # Non-string types should always pass (graceful handling)
            # except for bytes which converts to string
            if not isinstance(value, str | bytes):
                assert result.passed is True


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBasedPythonCode:
    """Property-based tests generating valid Python-like code.

    Note: These tests use tempfile instead of pytest fixtures because
    hypothesis @given decorator doesn't work well with pytest fixtures
    as function parameters.
    """

    @given(
        st.lists(
            st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,10}", fullmatch=True),
            min_size=0,
            max_size=5,
        )
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_generated_class_names_no_crash(
        self,
        class_names: list[str],
    ) -> None:
        """Generated Python classes should be handled without crashing."""
        import tempfile

        classes = []
        for name in class_names:
            classes.append(f"class {name}:\n    pass\n")
        content = "\n".join(classes) if classes else "# empty"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            file_path = f.name

        try:
            for rule_class in [
                RuleNoDirectDispatch,
                RuleNoHandlerPublishing,
                RuleNoOrchestratorFSM,
            ]:
                rule = rule_class()
                result = rule.check(file_path)
                assert_result_is_valid(result)
        finally:
            Path(file_path).unlink(missing_ok=True)

    @given(
        st.lists(
            st.tuples(
                st.sampled_from(["Handler", "Orchestrator", "Service", "Manager"]),
                st.from_regex(r"[A-Z][a-z]{0,10}", fullmatch=True),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_generated_mixed_classes_no_crash(
        self,
        class_parts: list[tuple[str, str]],
    ) -> None:
        """Generated Handler/Orchestrator classes should be handled without crashing."""
        import tempfile

        classes = []
        for prefix, suffix in class_parts:
            name = f"{prefix}{suffix}"
            classes.append(f"""
class {name}:
    def __init__(self, container):
        self._container = container

    def process(self, event):
        return event
""")
        content = "\n".join(classes)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            file_path = f.name

        try:
            for rule_class in [
                RuleNoDirectDispatch,
                RuleNoHandlerPublishing,
                RuleNoOrchestratorFSM,
            ]:
                rule = rule_class()
                result = rule.check(file_path)
                assert_result_is_valid(result)
        finally:
            Path(file_path).unlink(missing_ok=True)


# =============================================================================
# Regression Tests: Known Edge Cases
# =============================================================================


class TestRegressionKnownEdgeCases:
    """Tests for known edge cases that caused issues in the past."""

    def test_handler_string_in_comment(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """'Handler' in comment should not trigger false positive."""
        content = """
# This class is not a Handler, it's a Service
class UserService:
    def process(self):
        # Handler pattern: delegate to runtime
        return self.runtime.dispatch(event)
"""
        file_path = create_temp_file("comment_handler.py", content)
        result = rule_no_direct_dispatch.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_handler_in_string_literal(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """'Handler' in string should not trigger false positive."""
        content = '''
class Docs:
    HELP = """
    The Handler class is responsible for...
    Handler.handle() processes events...
    """

    def get_help(self):
        return "Use Handler.handle() for processing"
'''
        file_path = create_temp_file("string_handler.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_orchestrator_in_variable_name(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """'Orchestrator' in variable name should not trigger false positive."""
        content = """
class Coordinator:
    def setup(self):
        orchestrator_name = "main"
        self._orchestrator_count = 5
        return orchestrator_name
"""
        file_path = create_temp_file("var_orchestrator.py", content)
        result = rule_no_orchestrator_fsm.check(str(file_path))
        assert_result_is_valid(result)
        assert result.passed is True

    def test_partial_class_name_match(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Class name partially matching 'Handler' should still be caught."""
        content = """
class MyHandlerBase:
    def __init__(self, event_bus):
        self._bus = event_bus  # VIOLATION

class EventHandlerManager:
    def handle(self, event):
        return event  # OK - no bus access
"""
        file_path = create_temp_file("partial_match.py", content)
        result = rule_no_handler_publishing.check(str(file_path))
        assert_result_is_valid(result)
        # Should detect violation in MyHandlerBase
        assert result.passed is False

    def test_none_as_file_path(
        self,
        rule_no_direct_dispatch: RuleNoDirectDispatch,
    ) -> None:
        """None passed as file path should be handled gracefully."""
        result = rule_no_direct_dispatch.check(None)
        assert_result_is_valid(result)
        assert result.passed is True

    def test_empty_string_as_file_path(
        self,
        rule_no_handler_publishing: RuleNoHandlerPublishing,
    ) -> None:
        """Empty string as file path should be handled gracefully."""
        result = rule_no_handler_publishing.check("")
        assert_result_is_valid(result)
        assert result.passed is True

    def test_special_chars_in_path(
        self,
        rule_no_orchestrator_fsm: RuleNoOrchestratorFSM,
    ) -> None:
        """Path with special characters should be handled gracefully."""
        result = rule_no_orchestrator_fsm.check("/path/with spaces/and$pecial#chars.py")
        assert_result_is_valid(result)
        # Non-existent file should pass gracefully
        assert result.passed is True
