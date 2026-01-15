# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validator for ARCH-001: No Direct Handler Dispatch.

This validator ensures that handlers are dispatched through the runtime,
not called directly. Direct handler calls bypass the runtime's event
tracking, circuit breaking, and other cross-cutting concerns.

Related:
    - Ticket: OMN-1099 (Architecture Validator)
    - PR: #124 (Protocol-Compliant Rule Classes)
    - Rule: ARCH-001 (No Direct Handler Dispatch)

Detection Patterns:
    1. **Handler variable tracking**: Variables assigned from Handler() instantiation
    2. **Attribute access**: Attributes containing 'handler' (e.g., self._handler)
    3. **Inline instantiation**: Handler().handle() calls

Example Violations::

    # VIOLATION: Direct handler instantiation and call
    handler = MyHandler(container)
    result = handler.handle(event)  # Direct dispatch detected

    # VIOLATION: Attribute-based handler call
    self._handler.handle(event)  # Handler via attribute

    # VIOLATION: Inline handler dispatch
    MyHandler(container).handle(event)  # Instantiate and call

Allowed Patterns::

    # OK: Dispatch through runtime (proper pattern)
    self.runtime.dispatch(event)

    # OK: Test files are exempt (needed for unit testing)
    def test_handler():
        handler.handle(test_event)  # Allowed in test files

    # OK: Handler calling its own handle() method
    class MyHandler:
        def process(self, event):
            return self.handle(event)  # self.handle() is allowed
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.nodes.architecture_validator.mixins import MixinFilePathRule
from omnibase_infra.nodes.architecture_validator.models.model_architecture_violation import (
    ModelArchitectureViolation,
)
from omnibase_infra.nodes.architecture_validator.models.model_validation_result import (
    ModelFileValidationResult,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.architecture_validator.models import ModelRuleCheckResult

RULE_ID = "ARCH-001"
RULE_NAME = "No Direct Handler Dispatch"


class DirectDispatchVisitor(ast.NodeVisitor):
    """AST visitor to detect direct handler dispatch patterns.

    This visitor tracks:
    1. Variables assigned from Handler class instantiation
    2. Attribute access where the attribute name contains 'handler'
    3. Inline Handler().handle() calls

    It then flags any .handle() calls on these patterns as violations.
    """

    def __init__(self, file_path: str) -> None:
        """Initialize the visitor.

        Args:
            file_path: Path to the file being analyzed.
        """
        self.file_path = file_path
        self.violations: list[ModelArchitectureViolation] = []
        self._handler_variables: set[str] = set()
        self._current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track current class context for self.handle() exemption.

        Args:
            node: Class definition AST node.
        """
        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track handler variable assignments.

        When we see `handler = HandlerSomething()`, we record 'handler'
        as a handler variable so we can flag `handler.handle()` later.

        Args:
            node: Assignment AST node.
        """
        if isinstance(node.value, ast.Call):
            func_name = self._get_name(node.value.func)
            if func_name and "Handler" in func_name:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._handler_variables.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect .handle() calls on handler instances.

        Args:
            node: Function call AST node.
        """
        if isinstance(node.func, ast.Attribute) and node.func.attr == "handle":
            caller = node.func.value

            # Skip self.handle() within handler classes themselves
            if isinstance(caller, ast.Name) and caller.id == "self":
                if self._current_class and "Handler" in self._current_class:
                    self.generic_visit(node)
                    return

            # Check if caller is a tracked handler variable
            if isinstance(caller, ast.Name):
                caller_name = caller.id
                if (
                    caller_name in self._handler_variables
                    or "handler" in caller_name.lower()
                ):
                    self._add_violation(node)

            # Check for attribute access like self._handler.handle() or self.handler.handle()
            elif isinstance(caller, ast.Attribute):
                attr_name = caller.attr
                if "handler" in attr_name.lower():
                    self._add_violation(node)

            # Check for inline instantiation: Handler().handle()
            elif isinstance(caller, ast.Call):
                func_name = self._get_name(caller.func)
                if func_name and "Handler" in func_name:
                    self._add_violation(node)

        self.generic_visit(node)

    def _get_name(self, node: ast.expr) -> str | None:
        """Extract name from AST node.

        Args:
            node: AST expression node.

        Returns:
            The name if extractable, None otherwise.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _add_violation(self, node: ast.Call) -> None:
        """Add a violation for direct dispatch.

        Args:
            node: The Call AST node representing the violation.
        """
        location = f"{self.file_path}:{node.lineno}"
        self.violations.append(
            ModelArchitectureViolation(
                rule_id=RULE_ID,
                rule_name=RULE_NAME,
                severity=EnumValidationSeverity.WARNING,
                target_type="handler",
                target_name=self._current_class or "unknown",
                message="Direct handler dispatch detected. Handlers must be invoked through runtime.",
                location=location,
                suggestion="Use runtime.dispatch(event) instead of handler.handle(event)",
            )
        )


def _is_test_file(file_path: str) -> bool:
    """Check if file is a test file (exempt from rule).

    Test files are exempt because they need to test handlers directly.

    Args:
        file_path: Path to check.

    Returns:
        True if this is a test file, False otherwise.
    """
    path = Path(file_path)
    name = path.name
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or "/tests/" in str(path)
    )


def validate_no_direct_dispatch(file_path: str) -> ModelFileValidationResult:
    """Validate that handlers are not dispatched directly.

    This function checks a Python file for direct handler dispatch patterns
    that violate ARCH-001. Direct handler calls should be replaced with
    runtime.dispatch() calls.

    Args:
        file_path: Path to the Python file to validate.

    Returns:
        ModelArchitectureValidationResult with validation status and any violations.

    Note:
        Test files (files starting with "test_" or ending with "_test.py",
        conftest.py, or files in /tests/ directories) are exempt from this
        rule as they need to test handlers directly.
    """
    path = Path(file_path)

    # Test files are exempt
    if _is_test_file(file_path):
        return ModelFileValidationResult(
            valid=True,
            violations=[],
            files_checked=1,
            rules_checked=[RULE_ID],
        )

    # Non-existent or non-Python files
    if not path.exists() or path.suffix != ".py":
        return ModelFileValidationResult(
            valid=True,
            violations=[],
            files_checked=0,
            rules_checked=[RULE_ID],
        )

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except SyntaxError as e:
        # Return WARNING violation for syntax error
        location = f"{file_path}:{e.lineno}" if e.lineno else file_path
        return ModelFileValidationResult(
            valid=True,  # Still valid (not a rule violation), but with warning
            violations=[
                ModelArchitectureViolation(
                    rule_id=RULE_ID,
                    rule_name=RULE_NAME,
                    severity=EnumValidationSeverity.WARNING,
                    target_type="file",
                    target_name=Path(file_path).name,
                    message=f"File has syntax error and could not be validated: {e.msg}",
                    location=location,
                    suggestion="Fix the syntax error to enable architecture validation",
                )
            ],
            files_checked=1,
            rules_checked=[RULE_ID],
        )
    except (PermissionError, OSError) as e:
        # Return WARNING violation for file I/O errors
        return ModelFileValidationResult(
            valid=True,  # Still valid (not a rule violation), but with warning
            violations=[
                ModelArchitectureViolation(
                    rule_id=RULE_ID,
                    rule_name=RULE_NAME,
                    severity=EnumValidationSeverity.WARNING,
                    target_type="file",
                    target_name=Path(file_path).name,
                    message=f"File could not be read: {e}",
                    location=file_path,
                    suggestion="Ensure file is readable and has correct permissions",
                )
            ],
            files_checked=1,
            rules_checked=[RULE_ID],
        )
    except UnicodeDecodeError as e:
        # Return WARNING violation for encoding errors
        return ModelFileValidationResult(
            valid=True,  # Still valid (not a rule violation), but with warning
            violations=[
                ModelArchitectureViolation(
                    rule_id=RULE_ID,
                    rule_name=RULE_NAME,
                    severity=EnumValidationSeverity.WARNING,
                    target_type="file",
                    target_name=Path(file_path).name,
                    message=f"File has encoding error and could not be validated: {e.reason}",
                    location=file_path,
                    suggestion="Ensure file is valid UTF-8 encoded",
                )
            ],
            files_checked=1,
            rules_checked=[RULE_ID],
        )

    visitor = DirectDispatchVisitor(file_path)
    visitor.visit(tree)

    return ModelFileValidationResult(
        valid=len(visitor.violations) == 0,
        violations=visitor.violations,
        files_checked=1,
        rules_checked=[RULE_ID],
    )


class RuleNoDirectDispatch(MixinFilePathRule):
    """Protocol-compliant rule: No direct handler dispatch.

    This class wraps the file-based validator to implement ProtocolArchitectureRule,
    enabling use with NodeArchitectureValidatorCompute.

    Thread Safety:
        This rule is stateless and safe for concurrent use.
    """

    @property
    def rule_id(self) -> str:
        """Return the canonical rule ID matching contract.yaml."""
        return RULE_ID

    @property
    def name(self) -> str:
        """Return human-readable rule name."""
        return RULE_NAME

    @property
    def description(self) -> str:
        """Return detailed rule description."""
        return (
            "Handlers must be dispatched through the runtime, not called directly. "
            "Direct handler calls bypass runtime event tracking and circuit breaking."
        )

    @property
    def severity(self) -> EnumValidationSeverity:
        """Return severity level for violations of this rule.

        Note: Contract specifies WARNING severity for ARCH-001. This is appropriate
        because AST-based pattern detection may produce false positives for
        legitimate patterns like test mocks or debugging code. Using WARNING
        allows non-blocking validation while still flagging potential issues
        for review.
        """
        return EnumValidationSeverity.WARNING

    def check(self, target: object) -> ModelRuleCheckResult:
        """Check target against this rule.

        Args:
            target: Target to validate. If a string, treated as file path.
                   Other types return skipped=True with reason.

        Returns:
            ModelRuleCheckResult indicating pass/fail with details.

        Note:
            When multiple violations are found, only the first violation's
            message and location are returned. The total count is available
            in ``details["total_violations"]``. This fail-fast behavior is
            intentional - fix the first violation and re-run to find others.
        """
        from omnibase_infra.nodes.architecture_validator.models import (
            ModelRuleCheckResult,
        )

        file_path = self._extract_file_path(target)
        if file_path is None:
            return ModelRuleCheckResult(
                passed=True,
                rule_id=self.rule_id,
                skipped=True,
                reason="Target is not a valid file path",
            )

        # Delegate to existing file-based validator
        result = validate_no_direct_dispatch(file_path)

        if result.valid:
            return ModelRuleCheckResult(passed=True, rule_id=self.rule_id)

        # Convert first violation to ModelRuleCheckResult
        if result.violations:
            violation = result.violations[0]
            return ModelRuleCheckResult(
                passed=False,
                rule_id=self.rule_id,
                message=violation.message,
                details={
                    "target_name": violation.target_name,
                    "target_type": violation.target_type,
                    "location": violation.location,
                    "suggestion": violation.suggestion,
                    "total_violations": len(result.violations),
                },
            )

        return ModelRuleCheckResult(
            passed=False,
            rule_id=self.rule_id,
            message="Direct handler dispatch violation detected",
        )


__all__ = ["validate_no_direct_dispatch", "RuleNoDirectDispatch"]
