# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validator for ARCH-002: No Handler Publishing.

This validator uses AST analysis to detect handlers that directly publish
events to the event bus. Only orchestrators may publish events.

Detection patterns:
    - Handler class with event_bus, bus, or publisher in __init__ signature
    - Handler class with _bus, _event_bus, _publisher attributes
    - Handler class calling publish(), emit(), or send_event() methods

Handler identification:
    - Classes with "Handler" in the name (case sensitive)
    - Classes with "Orchestrator" in the name are NOT handlers

Example violations:
    >>> # BAD: Handler with event bus access
    >>> class HandlerBad:
    ...     def __init__(self, container, event_bus):
    ...         self._bus = event_bus
    ...     def handle(self, event):
    ...         self._bus.publish(SomeEvent())  # VIOLATION

    >>> # GOOD: Handler returns event for orchestrator to publish
    >>> class HandlerGood:
    ...     def handle(self, event):
    ...         return ModelEventEnvelope(payload=SomeEvent())  # OK
"""

from __future__ import annotations

import ast
from pathlib import Path

from omnibase_infra.nodes.architecture_validator.models import (
    EnumViolationSeverity,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
)

RULE_ID = "ARCH-002"
RULE_NAME = "No Handler Publishing"

# Forbidden patterns in constructor parameters
FORBIDDEN_PARAMS = {"event_bus", "bus", "publisher"}

# Forbidden attribute names (assignment targets)
FORBIDDEN_ATTRS = {"_bus", "_event_bus", "_publisher", "event_bus", "publisher"}

# Forbidden method names (publish-like calls)
FORBIDDEN_METHODS = {"publish", "emit", "send_event"}


class PublishingConstraintVisitor(ast.NodeVisitor):
    """AST visitor to detect handler publishing patterns.

    This visitor traverses Python AST to find handler classes that
    violate the ARCH-002 rule by having direct event bus access or
    calling publish-like methods.

    Attributes:
        file_path: Path to the file being analyzed.
        violations: List of violations found during traversal.
    """

    def __init__(self, file_path: str) -> None:
        """Initialize the visitor.

        Args:
            file_path: Path to the file being analyzed.
        """
        self.file_path = file_path
        self.violations: list[ModelArchitectureViolation] = []
        self._in_handler_class = False
        self._current_class_name: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track handler class context.

        Identifies handler classes by naming convention (contains "Handler"
        but not "Orchestrator") and marks context for nested visits.

        Args:
            node: AST ClassDef node to visit.
        """
        # Determine if this is a handler class
        is_handler = "Handler" in node.name
        is_orchestrator = "Orchestrator" in node.name

        # Only flag handlers, not orchestrators
        # Orchestrators ARE allowed to have event bus access
        old_in_handler = self._in_handler_class
        old_class_name = self._current_class_name

        self._in_handler_class = is_handler and not is_orchestrator
        self._current_class_name = node.name

        # Visit child nodes
        self.generic_visit(node)

        # Restore context
        self._in_handler_class = old_in_handler
        self._current_class_name = old_class_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions for forbidden parameters.

        Only checks __init__ methods within handler classes for
        forbidden event bus parameters.

        Args:
            node: AST FunctionDef node to visit.
        """
        if self._in_handler_class and node.name == "__init__":
            # Check all arguments for forbidden parameters
            for arg in node.args.args:
                if arg.arg in FORBIDDEN_PARAMS:
                    self._add_violation(
                        node,
                        f"Handler '{self._current_class_name}' constructor has "
                        f"forbidden parameter '{arg.arg}'",
                        "Handlers must not receive event bus. "
                        "Return events for orchestrator to publish.",
                    )

        # Continue visiting nested nodes
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async function definitions for forbidden parameters.

        Same logic as visit_FunctionDef but for async methods.

        Args:
            node: AST AsyncFunctionDef node to visit.
        """
        if self._in_handler_class and node.name == "__init__":
            for arg in node.args.args:
                if arg.arg in FORBIDDEN_PARAMS:
                    self._add_violation(
                        node,
                        f"Handler '{self._current_class_name}' constructor has "
                        f"forbidden parameter '{arg.arg}'",
                        "Handlers must not receive event bus. "
                        "Return events for orchestrator to publish.",
                    )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for forbidden attribute assignments.

        Detects assignments to attributes like _bus, _event_bus, _publisher
        within handler classes.

        Args:
            node: AST Assign node to visit.
        """
        if self._in_handler_class:
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    if target.attr in FORBIDDEN_ATTRS:
                        self._add_violation(
                            node,
                            f"Handler '{self._current_class_name}' has "
                            f"forbidden attribute '{target.attr}'",
                            "Remove event bus attribute. "
                            "Return events for orchestrator to publish.",
                        )

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check for forbidden annotated attribute assignments.

        Same logic as visit_Assign but for annotated assignments (e.g., self._bus: EventBus = ...).

        Args:
            node: AST AnnAssign node to visit.
        """
        if self._in_handler_class:
            target = node.target
            if isinstance(target, ast.Attribute):
                if target.attr in FORBIDDEN_ATTRS:
                    self._add_violation(
                        node,
                        f"Handler '{self._current_class_name}' has "
                        f"forbidden attribute '{target.attr}'",
                        "Remove event bus attribute. "
                        "Return events for orchestrator to publish.",
                    )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for forbidden publish method calls.

        Detects calls to methods like publish(), emit(), send_event()
        within handler classes.

        Args:
            node: AST Call node to visit.
        """
        if self._in_handler_class:
            # Check for method calls like self._bus.publish() or self.emit()
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_METHODS:
                    self._add_violation(
                        node,
                        f"Handler '{self._current_class_name}' calls "
                        f"forbidden method '{node.func.attr}()'",
                        "Return events instead of publishing directly. "
                        "The orchestrator will handle publishing.",
                    )

        self.generic_visit(node)

    def _add_violation(
        self,
        node: ast.AST,
        message: str,
        suggestion: str,
    ) -> None:
        """Add a violation to the list.

        Args:
            node: AST node where violation occurred.
            message: Description of the violation.
            suggestion: How to fix the violation.
        """
        self.violations.append(
            ModelArchitectureViolation(
                rule_id=RULE_ID,
                rule_name=RULE_NAME,
                severity=EnumViolationSeverity.ERROR,
                file_path=self.file_path,
                line_number=getattr(node, "lineno", None),
                message=message,
                suggestion=suggestion,
            )
        )


def validate_no_handler_publishing(file_path: str) -> ModelArchitectureValidationResult:
    """Validate that handlers do not publish events directly.

    This function parses a Python source file and uses AST analysis to
    detect handler classes that violate the ARCH-002 rule.

    Args:
        file_path: Path to the Python source file to validate.

    Returns:
        ModelArchitectureValidationResult with validation status and any violations.

    Example:
        >>> result = validate_no_handler_publishing("path/to/handler.py")
        >>> if not result.valid:
        ...     for v in result.violations:
        ...         print(f"{v.rule_id}: {v.message}")
    """
    path = Path(file_path)

    # Handle non-existent files or non-Python files
    if not path.exists() or path.suffix != ".py":
        return ModelArchitectureValidationResult(
            valid=True,
            violations=[],
            files_checked=0,
            rules_checked=[RULE_ID],
        )

    # Read and parse the source file
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

    # Analyze the AST
    visitor = PublishingConstraintVisitor(file_path)
    visitor.visit(tree)

    return ModelArchitectureValidationResult(
        valid=len(visitor.violations) == 0,
        violations=visitor.violations,
        files_checked=1,
        rules_checked=[RULE_ID],
    )


__all__ = ["validate_no_handler_publishing"]
