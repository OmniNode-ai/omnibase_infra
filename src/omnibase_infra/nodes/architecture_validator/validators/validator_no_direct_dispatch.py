# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validator for ARCH-001: No Direct Handler Dispatch.

This validator ensures that handlers are dispatched through the runtime,
not called directly. Direct handler calls bypass the runtime's event
tracking, circuit breaking, and other cross-cutting concerns.

Related:
    - Ticket: OMN-1099 (Architecture Validator)
    - Rule: ARCH-001 (No Direct Handler Dispatch)

Example Violations:
    # VIOLATION: Direct handler instantiation and call
    handler = MyHandler(container)
    result = handler.handle(event)

    # VIOLATION: Inline handler dispatch
    handler.handle(event)

Allowed Patterns:
    # OK: Dispatch through runtime
    self.runtime.dispatch(event)

    # OK: Test files are exempt
    def test_handler():
        handler.handle(test_event)  # Allowed in tests
"""

from __future__ import annotations

import ast
from pathlib import Path

from omnibase_infra.nodes.architecture_validator.models import (
    EnumViolationSeverity,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
)

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
        self.violations.append(
            ModelArchitectureViolation(
                rule_id=RULE_ID,
                rule_name=RULE_NAME,
                severity=EnumViolationSeverity.ERROR,
                file_path=self.file_path,
                line_number=node.lineno,
                message="Direct handler dispatch detected. Handlers must be invoked through runtime.",
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


def validate_no_direct_dispatch(file_path: str) -> ModelArchitectureValidationResult:
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
        return ModelArchitectureValidationResult(
            valid=True,
            violations=[],
            files_checked=1,
            rules_checked=[RULE_ID],
        )

    # Non-existent or non-Python files
    if not path.exists() or path.suffix != ".py":
        return ModelArchitectureValidationResult(
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
        return ModelArchitectureValidationResult(
            valid=True,  # Still valid (not a rule violation), but with warning
            violations=[
                ModelArchitectureViolation(
                    rule_id=RULE_ID,
                    rule_name=RULE_NAME,
                    severity=EnumViolationSeverity.WARNING,
                    file_path=file_path,
                    line_number=e.lineno,
                    message=f"File has syntax error and could not be validated: {e.msg}",
                    suggestion="Fix the syntax error to enable architecture validation",
                )
            ],
            files_checked=1,
            rules_checked=[RULE_ID],
        )

    visitor = DirectDispatchVisitor(file_path)
    visitor.visit(tree)

    return ModelArchitectureValidationResult(
        valid=len(visitor.violations) == 0,
        violations=visitor.violations,
        files_checked=1,
        rules_checked=[RULE_ID],
    )


__all__ = ["validate_no_direct_dispatch"]
