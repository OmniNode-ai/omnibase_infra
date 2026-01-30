# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the declarative node validator.

Tests the detection of imperative patterns in node.py files.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_core.models.common.model_validation_result import ModelValidationResult
from omnibase_core.models.contracts.subcontracts.model_validator_rule import (
    ModelValidatorRule,
)
from omnibase_core.models.contracts.subcontracts.model_validator_subcontract import (
    ModelValidatorSubcontract,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums.enum_declarative_node_violation import (
    EnumDeclarativeNodeViolation,
)
from omnibase_infra.validation.validator_declarative_node import (
    ValidatorDeclarativeNode,
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


def _create_custom_contract(
    *,
    rules: list[ModelValidatorRule] | None = None,
    target_patterns: list[str] | None = None,
) -> ModelValidatorSubcontract:
    """Create a custom contract for testing.

    Args:
        rules: List of validation rules. If None, creates default rules.
        target_patterns: File patterns to target. Defaults to ["**/node.py"].

    Returns:
        A ModelValidatorSubcontract instance for testing.
    """
    if target_patterns is None:
        target_patterns = ["**/node.py", "node.py"]

    if rules is None:
        # Create all default rules as enabled
        rules = [
            ModelValidatorRule(
                rule_id="DECL-001",
                description="No custom methods",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-002",
                description="No custom properties",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-003",
                description="No init custom logic",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-004",
                description="No instance variables",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-005",
                description="No class variables",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-006",
                description="Syntax errors",
                enabled=True,
            ),
        ]

    return ModelValidatorSubcontract(
        version=ModelSemVer(major=1, minor=0, patch=0),
        validator_id="declarative_node",
        validator_name="Test Declarative Node Validator",
        validator_description="Test validator for declarative nodes",
        target_patterns=target_patterns,
        rules=rules,
        suppression_comments=["# ONEX_EXCLUDE: declarative_node"],
    )


class TestValidatorDeclarativeNodeClass:
    """Tests for ValidatorDeclarativeNode class API."""

    def test_validator_class_validate_method(self, tmp_path: Path) -> None:
        """ValidatorDeclarativeNode.validate() should work on directories."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Valid declarative node.'''
            pass
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        # Create validator with custom contract (default rules)
        contract = _create_custom_contract()
        validator = ValidatorDeclarativeNode(contract=contract)

        # Call validate() on the directory
        result = validator.validate(tmp_path / "nodes")

        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_validator_class_returns_model_validation_result(
        self, tmp_path: Path
    ) -> None:
        """Validator should return ModelValidationResult with proper structure."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Valid declarative node.'''
            pass
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        contract = _create_custom_contract()
        validator = ValidatorDeclarativeNode(contract=contract)
        result = validator.validate(tmp_path / "nodes")

        # Verify the result is the correct type
        assert isinstance(result, ModelValidationResult)
        # Verify result has expected attributes
        assert hasattr(result, "is_valid")
        assert hasattr(result, "issues")
        assert hasattr(result, "summary")
        assert hasattr(result, "metadata")
        # Verify metadata is populated
        assert result.metadata is not None
        assert result.metadata.validation_type == "declarative_node"
        assert result.metadata.files_processed >= 1

    def test_validator_class_with_violations_returns_invalid(
        self, tmp_path: Path
    ) -> None:
        """Validator should return is_valid=False when violations exist."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with custom method.'''

            def custom_method(self):
                return "violation"
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        contract = _create_custom_contract()
        validator = ValidatorDeclarativeNode(contract=contract)
        result = validator.validate(tmp_path / "nodes")

        assert result.is_valid is False
        assert len(result.issues) >= 1
        # Verify the issue has the expected structure
        issue = result.issues[0]
        assert issue.message is not None
        assert "custom_method" in issue.message
        assert issue.code == "DECL-001"

    def test_validator_class_respects_contract_rules_disabled(
        self, tmp_path: Path
    ) -> None:
        """Validator should skip violations for disabled rules."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with custom method - but rule is disabled.'''

            def custom_method(self):
                return "should not be reported"
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        # Create contract with DECL-001 (custom methods) DISABLED
        rules = [
            ModelValidatorRule(
                rule_id="DECL-001",
                description="No custom methods",
                enabled=False,  # DISABLED
            ),
            ModelValidatorRule(
                rule_id="DECL-002",
                description="No custom properties",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-003",
                description="No init custom logic",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-004",
                description="No instance variables",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-005",
                description="No class variables",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-006",
                description="Syntax errors",
                enabled=True,
            ),
        ]
        contract = _create_custom_contract(rules=rules)
        validator = ValidatorDeclarativeNode(contract=contract)
        result = validator.validate(tmp_path / "nodes")

        # Should pass because custom method rule is disabled
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_validator_class_validate_file_method(self, tmp_path: Path) -> None:
        """ValidatorDeclarativeNode.validate_file() should work on single files."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            pass
        """
        filepath = _create_test_file(tmp_path, code)

        contract = _create_custom_contract()
        validator = ValidatorDeclarativeNode(contract=contract)
        result = validator.validate_file(filepath)

        assert isinstance(result, ModelValidationResult)
        assert result.is_valid is True


class TestMultipleNodeClasses:
    """Tests for files with multiple node classes."""

    def test_multiple_node_classes_all_valid(self, tmp_path: Path) -> None:
        """File with multiple valid node classes should pass."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect
        from omnibase_core.nodes.node_compute import NodeCompute

        class FirstNode(NodeEffect):
            '''First valid declarative node.'''
            pass

        class SecondNode(NodeCompute):
            '''Second valid declarative node.'''
            pass
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        assert len(violations) == 0

    def test_multiple_node_classes_mixed_compliance(self, tmp_path: Path) -> None:
        """File with valid and invalid node classes should report only invalid."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect
        from omnibase_core.nodes.node_compute import NodeCompute

        class ValidNode(NodeEffect):
            '''Valid declarative node - no custom logic.'''
            pass

        class InvalidNode(NodeCompute):
            '''Invalid node with custom method.'''

            def custom_method(self):
                return "violation"
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        # Should only report violations for InvalidNode
        assert len(violations) == 1
        assert violations[0].node_class_name == "InvalidNode"
        assert violations[0].method_name == "custom_method"
        assert (
            violations[0].violation_type == EnumDeclarativeNodeViolation.CUSTOM_METHOD
        )

    def test_multiple_node_classes_multiple_invalid(self, tmp_path: Path) -> None:
        """File with multiple invalid node classes should report all violations."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect
        from omnibase_core.nodes.node_compute import NodeCompute

        class FirstInvalid(NodeEffect):
            '''First invalid node.'''

            def method_one(self):
                return "first violation"

        class SecondInvalid(NodeCompute):
            '''Second invalid node.'''

            CLASS_VAR = "second violation"
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        # Should report violations for both classes
        assert len(violations) == 2
        class_names = {v.node_class_name for v in violations}
        assert "FirstInvalid" in class_names
        assert "SecondInvalid" in class_names

    def test_multiple_node_classes_one_exempted(self, tmp_path: Path) -> None:
        """Exempted class should not generate violations even with multiple classes."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect
        from omnibase_core.nodes.node_compute import NodeCompute

        class ValidNode(NodeEffect):
            '''Valid declarative node.'''
            pass

        # ONEX_EXCLUDE: declarative_node
        class ExemptedNode(NodeCompute):
            '''Exempted - intentionally imperative.'''

            def custom_method(self):
                return "exempted violation"
        """
        filepath = _create_test_file(tmp_path, code)

        violations = validate_declarative_node_in_file(filepath)

        # Should have no violations - ValidNode is clean, ExemptedNode is exempted
        assert len(violations) == 0

    def test_multiple_node_classes_via_class_api(self, tmp_path: Path) -> None:
        """ValidatorDeclarativeNode class should handle multiple classes correctly."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect
        from omnibase_core.nodes.node_reducer import NodeReducer

        class CleanNode(NodeEffect):
            '''Clean node.'''
            pass

        class DirtyNode(NodeReducer):
            '''Dirty node with property.'''

            @property
            def bad_property(self):
                return "violation"
        """
        node_dir = tmp_path / "nodes" / "multi_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        contract = _create_custom_contract()
        validator = ValidatorDeclarativeNode(contract=contract)
        result = validator.validate(tmp_path / "nodes")

        assert result.is_valid is False
        assert len(result.issues) == 1
        # Verify the issue is for DirtyNode
        assert "DirtyNode" in result.issues[0].message
        assert "bad_property" in result.issues[0].message


class TestContractRuleConfiguration:
    """Tests for contract-based rule configuration."""

    def test_disabled_property_rule_allows_properties(self, tmp_path: Path) -> None:
        """Disabling DECL-002 should allow @property decorators."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with property - but rule is disabled.'''

            @property
            def my_property(self):
                return "allowed because rule disabled"
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        # Create contract with DECL-002 (properties) DISABLED
        rules = [
            ModelValidatorRule(
                rule_id="DECL-001",
                description="No custom methods",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-002",
                description="No custom properties",
                enabled=False,  # DISABLED
            ),
            ModelValidatorRule(
                rule_id="DECL-003",
                description="No init custom logic",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-004",
                description="No instance variables",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-005",
                description="No class variables",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-006",
                description="Syntax errors",
                enabled=True,
            ),
        ]
        contract = _create_custom_contract(rules=rules)
        validator = ValidatorDeclarativeNode(contract=contract)
        result = validator.validate(tmp_path / "nodes")

        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_disabled_class_variable_rule(self, tmp_path: Path) -> None:
        """Disabling DECL-005 should allow class variables."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node with class variable - but rule is disabled.'''

            CLASS_CONSTANT = "allowed"
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        # Create contract with DECL-005 (class variables) DISABLED
        rules = [
            ModelValidatorRule(
                rule_id="DECL-001",
                description="No custom methods",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-002",
                description="No custom properties",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-003",
                description="No init custom logic",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-004",
                description="No instance variables",
                enabled=True,
            ),
            ModelValidatorRule(
                rule_id="DECL-005",
                description="No class variables",
                enabled=False,  # DISABLED
            ),
            ModelValidatorRule(
                rule_id="DECL-006",
                description="Syntax errors",
                enabled=True,
            ),
        ]
        contract = _create_custom_contract(rules=rules)
        validator = ValidatorDeclarativeNode(contract=contract)
        result = validator.validate(tmp_path / "nodes")

        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_all_rules_disabled_allows_everything(self, tmp_path: Path) -> None:
        """Disabling all rules should allow any node structure."""
        code = """
        from omnibase_core.nodes.node_effect import NodeEffect

        class MyNodeEffect(NodeEffect):
            '''Node that would normally have multiple violations.'''

            CLASS_VAR = "violation 1"

            def __init__(self, container):
                super().__init__(container)
                self._instance_var = "violation 2"

            def custom_method(self):
                return "violation 3"

            @property
            def custom_prop(self):
                return "violation 4"
        """
        node_dir = tmp_path / "nodes" / "test_node"
        node_dir.mkdir(parents=True)
        (node_dir / "node.py").write_text(dedent(code))

        # Create contract with ALL rules DISABLED
        rules = [
            ModelValidatorRule(
                rule_id="DECL-001",
                description="No custom methods",
                enabled=False,
            ),
            ModelValidatorRule(
                rule_id="DECL-002",
                description="No custom properties",
                enabled=False,
            ),
            ModelValidatorRule(
                rule_id="DECL-003",
                description="No init custom logic",
                enabled=False,
            ),
            ModelValidatorRule(
                rule_id="DECL-004",
                description="No instance variables",
                enabled=False,
            ),
            ModelValidatorRule(
                rule_id="DECL-005",
                description="No class variables",
                enabled=False,
            ),
            ModelValidatorRule(
                rule_id="DECL-006",
                description="Syntax errors",
                enabled=False,
            ),
        ]
        contract = _create_custom_contract(rules=rules)
        validator = ValidatorDeclarativeNode(contract=contract)
        result = validator.validate(tmp_path / "nodes")

        # All rules disabled, so should pass
        assert result.is_valid is True
        assert len(result.issues) == 0


__all__ = [
    "TestDeclarativeNodePass",
    "TestDeclarativeNodeViolations",
    "TestDeclarativeNodeCIResult",
    "TestDeclarativeNodeEdgeCases",
    "TestValidatorDeclarativeNodeClass",
    "TestMultipleNodeClasses",
    "TestContractRuleConfiguration",
]
