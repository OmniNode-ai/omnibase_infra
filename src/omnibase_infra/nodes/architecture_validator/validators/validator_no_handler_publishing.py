# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validator for ARCH-002: No Handler Publishing.

This validator uses AST analysis to detect handlers that directly publish
events to the event bus. Only orchestrators may publish events. This enforces
the ONEX architectural principle that handlers are pure processors that
return results, while orchestrators own the event publishing lifecycle.

Related:
    - Ticket: OMN-1099 (Architecture Validator)
    - PR: #124 (Protocol-Compliant Rule Classes)
    - Rule: ARCH-002 (No Handler Publishing)
    - See also: docs/patterns/dispatcher_resilience.md

Detection Patterns:
    1. **Constructor parameters**: event_bus, bus, or publisher in __init__ signature
    2. **Instance attributes**: _bus, _event_bus, _publisher assigned in handler
    3. **Method calls**: publish(), emit(), or send_event() invocations

Handler Identification:
    - Classes with "Handler" in the name (case sensitive)
    - Classes with "Orchestrator" in the name are NOT handlers (excluded)

Example Violations::

    # VIOLATION: Handler with event bus in constructor
    class HandlerBad:
        def __init__(self, container, event_bus):  # Forbidden parameter
            self._bus = event_bus  # Forbidden attribute
        def handle(self, event):
            self._bus.publish(SomeEvent())  # Forbidden method call

    # VIOLATION: Handler calling publish directly
    class HandlerAlsoBad:
        def handle(self, event):
            self.emit(SomeEvent())  # Forbidden method

Allowed Patterns::

    # OK: Handler returns event envelope for orchestrator to publish
    class HandlerGood:
        def handle(self, event):
            return ModelEventEnvelope(payload=SomeEvent())  # Correct pattern

    # OK: Orchestrator with event bus (orchestrators ARE allowed)
    class OrchestratorGood:
        def __init__(self, container, event_bus):  # Allowed for orchestrators
            self._bus = event_bus
        def orchestrate(self, event):
            self._bus.publish(response)  # Allowed
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
        line_number = getattr(node, "lineno", None)
        location = f"{self.file_path}:{line_number}" if line_number else self.file_path
        self.violations.append(
            ModelArchitectureViolation(
                rule_id=RULE_ID,
                rule_name=RULE_NAME,
                severity=EnumValidationSeverity.ERROR,
                target_type="handler",
                target_name=self._current_class_name or "unknown",
                message=message,
                location=location,
                suggestion=suggestion,
            )
        )


def validate_no_handler_publishing(file_path: str) -> ModelFileValidationResult:
    """Validate that handlers do not publish events directly.

    This function parses a Python source file and uses AST analysis to
    detect handler classes that violate the ARCH-002 rule.

    Args:
        file_path: Path to the Python source file to validate.

    Returns:
        ModelFileValidationResult with:
            - valid=True if no publishing patterns found in handlers
            - valid=False with violations if publishing patterns detected

    Note:
        Orchestrator classes are EXEMPT from this rule - they are allowed
        to publish events. Only classes with "Handler" in the name (but not
        "Orchestrator") are validated.

    Example::

        result = validate_no_handler_publishing("path/to/handler.py")
        if not result.valid:
            for v in result.violations:
                print(f"{v.rule_id}: {v.message}")
    """
    path = Path(file_path)

    # Handle non-existent files or non-Python files
    if not path.exists() or path.suffix != ".py":
        return ModelFileValidationResult(
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

    # Analyze the AST
    visitor = PublishingConstraintVisitor(file_path)
    visitor.visit(tree)

    return ModelFileValidationResult(
        valid=len(visitor.violations) == 0,
        violations=visitor.violations,
        files_checked=1,
        rules_checked=[RULE_ID],
    )


class RuleNoHandlerPublishing(MixinFilePathRule):
    """Protocol-compliant rule: Handlers must not publish events directly.

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
            "Handlers must not have direct event bus access. Only orchestrators "
            "may publish events. Handlers should return events for orchestrators to publish."
        )

    @property
    def severity(self) -> EnumValidationSeverity:
        """Return severity level for violations of this rule."""
        return EnumValidationSeverity.ERROR

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
        result = validate_no_handler_publishing(file_path)

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
            message="Handler publishing violation detected",
        )


__all__ = ["validate_no_handler_publishing", "RuleNoHandlerPublishing"]
