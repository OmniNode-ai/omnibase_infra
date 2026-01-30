# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the declarative node validator.

Tests the detection of imperative patterns in node.py files.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.enums.enum_declarative_node_violation import (
    EnumDeclarativeNodeViolation,
)
from omnibase_infra.validation.validator_declarative_node import (
    validate_declarative_node_in_file,
    validate_declarative_nodes_ci,
)


def _create_test_file(tmp_path: Path, content: str, filename: str = "node.py") -> Path:
    """Create a test Python file in a nodes directory structure.

    Args:
        tmp_path: Pytest temporary directory.
        content: Python code content.
        filename: Name of the file to create.

    Returns:
        Path to the created file.
    """
    # Create nodes/test_node/node.py structure
    node_dir = tmp_path / "nodes" / "test_node"
    node_dir.mkdir(parents=True, exist_ok=True)
    filepath = node_dir / filename
    filepath.write_text(dedent(content), encoding="utf-8")
    return filepath


class TestDeclarativeNodePass:
    """Tests for valid declarative nodes (should pass)."""

    def test_pass_minimal_declarative_node(self, tmp_path: Path) -> None:
        """Minimal declarative node with only class definition should pass."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Declarative effect node.'''
            pass
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 0

    def test_pass_declarative_node_with_init(self, tmp_path: Path) -> None:
        """Declarative node with valid __init__ should pass."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Declarative effect node.'''

            def __init__(self, container):
                '''Initialize the node.'''
                super().__init__(container)
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 0

    def test_pass_declarative_orchestrator(self, tmp_path: Path) -> None:
        """Declarative orchestrator should pass."""
        code = """
        from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

        class MyOrchestrator(NodeOrchestrator):
            '''Declarative orchestrator - all behavior in contract.yaml.'''

            def __init__(self, container):
                super().__init__(container)
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 0

    def test_pass_module_level_function(self, tmp_path: Path) -> None:
        """Module-level helper functions should NOT trigger violations."""
        code = """
        from pathlib import Path
        from omnibase_core.nodes.node_effect import NodeEffect

        def _helper_function():
            '''Module-level helper - allowed.'''
            return Path(__file__).parent / "contract.yaml"

        class MyNodeEffect(NodeEffect):
            '''Declarative effect node.'''
            pass
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 0

    def test_pass_exempted_class(self, tmp_path: Path) -> None:
        """Class exempted with ONEX_EXCLUDE comment should pass."""
        code = """
        from omnibase_core.nodes.node_compute import NodeCompute

        # ONEX_EXCLUDE: declarative_node
        class MyComputeNode(NodeCompute):
            '''Intentionally imperative for legacy reasons.'''

            def compute(self, data):
                '''Custom compute logic.'''
                return data.upper()
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 0


class TestDeclarativeNodeViolations:
    """Tests for detecting declarative node violations."""

    def test_detect_custom_method(self, tmp_path: Path) -> None:
        """Custom method in node class should be detected."""
        code = """
        from omnibase_core.nodes.node_compute import NodeCompute

        class MyComputeNode(NodeCompute):
            '''Node with custom method.'''

            def compute(self, data):
                '''Custom compute logic - VIOLATION.'''
                return data.upper()
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 1
        assert (
            violations[0].violation_type == EnumDeclarativeNodeViolation.CUSTOM_METHOD
        )
        assert violations[0].method_name == "compute"
        assert violations[0].node_class_name == "MyComputeNode"

    def test_detect_property(self, tmp_path: Path) -> None:
        """Property in node class should be detected."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with property.'''

            @property
            def my_property(self):
                '''Custom property - VIOLATION.'''
                return "value"
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 1
        assert (
            violations[0].violation_type == EnumDeclarativeNodeViolation.CUSTOM_PROPERTY
        )
        assert violations[0].method_name == "my_property"

    def test_detect_instance_variable(self, tmp_path: Path) -> None:
        """Instance variable in __init__ should be detected."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with instance variable.'''

            def __init__(self, container):
                super().__init__(container)
                self._custom_var = "value"  # VIOLATION
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        # Should detect both INIT_CUSTOM_LOGIC and INSTANCE_VARIABLE
        violation_types = {v.violation_type for v in violations}
        assert EnumDeclarativeNodeViolation.INSTANCE_VARIABLE in violation_types

    def test_detect_init_custom_logic(self, tmp_path: Path) -> None:
        """__init__ with logic beyond super().__init__ should be detected."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with custom __init__ logic.'''

            def __init__(self, container):
                super().__init__(container)
                print("Custom initialization")  # VIOLATION
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        violation_types = {v.violation_type for v in violations}
        assert EnumDeclarativeNodeViolation.INIT_CUSTOM_LOGIC in violation_types

    def test_detect_class_variable(self, tmp_path: Path) -> None:
        """Class variable in node class should be detected."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with class variable.'''

            CLASS_CONSTANT = "value"  # VIOLATION
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 1
        assert (
            violations[0].violation_type == EnumDeclarativeNodeViolation.CLASS_VARIABLE
        )

    def test_detect_multiple_violations(self, tmp_path: Path) -> None:
        """Multiple violations in single class should all be detected."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with multiple violations.'''

            CLASS_VAR = "const"  # VIOLATION 1

            def __init__(self, container):
                super().__init__(container)
                self._var = "value"  # VIOLATION 2

            def custom_method(self):  # VIOLATION 3
                return "result"

            @property
            def my_prop(self):  # VIOLATION 4
                return self._var
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) >= 4


class TestDeclarativeNodeCIResult:
    """Tests for CI result model behavior."""

    def test_ci_result_passes_on_clean_nodes(self, tmp_path: Path) -> None:
        """CI result should pass when all nodes are declarative."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            pass
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        result = validate_declarative_nodes_ci(tmp_path / "nodes")

        assert result.passed is True
        assert bool(result) is True
        assert result.blocking_count == 0

    def test_ci_result_fails_on_imperative_nodes(self, tmp_path: Path) -> None:
        """CI result should fail when imperative nodes exist."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            def custom_method(self):
                return "violation"
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        result = validate_declarative_nodes_ci(tmp_path / "nodes")

        assert result.passed is False
        assert bool(result) is False
        assert result.blocking_count >= 1
        assert "MyNodeEffect" in result.imperative_nodes

    def test_ci_result_format_summary(self, tmp_path: Path) -> None:
        """CI result should have format_summary method."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            pass
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        result = validate_declarative_nodes_ci(tmp_path / "nodes")

        summary = result.format_summary()
        assert "Declarative Node Validation" in summary
        assert "PASSED" in summary


class TestDeclarativeNodeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_syntax_error_file(self, tmp_path: Path) -> None:
        """Files with syntax errors should report as syntax error violation."""
        code = """
        class Broken(
            # Missing closing paren
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumDeclarativeNodeViolation.SYNTAX_ERROR

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file should have no violations."""
        filepath = _create_test_file(tmp_path, "")

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 0

    def test_non_node_class(self, tmp_path: Path) -> None:
        """Classes not inheriting from Node* should be ignored."""
        code = """
        class SomeHelper:
            '''Helper class - not a node.'''

            def helper_method(self):
                return "this is fine"
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 0

    def test_async_method_detected(self, tmp_path: Path) -> None:
        """Async methods in node class should be detected."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            async def async_method(self):
                return "async violation"
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 1
        assert (
            violations[0].violation_type == EnumDeclarativeNodeViolation.CUSTOM_METHOD
        )
        assert violations[0].method_name == "async_method"


__all__ = [
    "TestDeclarativeNodePass",
    "TestDeclarativeNodeViolations",
    "TestDeclarativeNodeCIResult",
    "TestDeclarativeNodeEdgeCases",
]
