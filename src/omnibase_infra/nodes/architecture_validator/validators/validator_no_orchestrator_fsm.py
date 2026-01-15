# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validator for ARCH-003: No Workflow FSM in Orchestrators.

This validator ensures that orchestrators do not implement workflow FSM
(finite state machine) logic that should be owned by reducers. Orchestrators
are "reaction planners" that coordinate work, not state machine owners.

KEY CLARIFICATION from ticket OMN-1099:
    - Reducers MAY implement aggregate state machines (that's their purpose)
    - Orchestrators must NOT implement workflow FSMs duplicating reducer transitions
    - Orchestrators are "reaction planners", not state machine owners

Related:
    - Ticket: OMN-1099 (Architecture Validator - FSM Rule Clarified)
    - PR: #124 (Protocol-Compliant Rule Classes)
    - Rule: ARCH-003 (No Workflow FSM in Orchestrators)

Detection Patterns:
    1. **Class-level FSM**: STATES, TRANSITIONS, STATE_MACHINE, FSM,
       ALLOWED_TRANSITIONS
    2. **State prefix**: STATE_* constants (e.g., STATE_PENDING)
    3. **Instance attributes**: _state, _workflow_state, _current_step,
       _fsm_state, _current_state
    4. **FSM methods**: transition(), can_transition(), apply_transition(),
       get_current_state(), set_state()

Example Violations::

    # VIOLATION: Orchestrator with state transition table
    class OrchestratorOrder(NodeOrchestrator):
        STATES = ["pending", "processing", "completed"]
        TRANSITIONS = {"pending": ["processing"], ...}

    # VIOLATION: Orchestrator tracking workflow state
    class OrchestratorPayment(NodeOrchestrator):
        def __init__(self, container):
            self._state = "initial"
            self._current_step = 0

    # VIOLATION: Orchestrator with transition methods
    class OrchestratorShipping(NodeOrchestrator):
        def can_transition(self, from_state, to_state):
            ...
        def apply_transition(self, transition):
            ...

Allowed Patterns::

    # OK: Reducers with FSM (this is what reducers do)
    class ReducerOrder(NodeReducer):
        STATES = ["created", "processing", "completed"]  # Allowed
        def reduce(self, state, event):
            ...

    # OK: Orchestrator delegating to reducer
    class OrchestratorUser(NodeOrchestrator):
        def orchestrate(self, event):
            intents = self._reducer.reduce(state, event)
            return self.plan_reactions(intents)

    # OK: Event-driven coordination (no state ownership)
    class OrchestratorCheckout(NodeOrchestrator):
        def orchestrate(self, event):
            match event:
                case CartSubmitted():
                    return [ValidateCart(), CalculateTotal()]
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

RULE_ID = "ARCH-003"
RULE_NAME = "No Workflow FSM in Orchestrators"

# FSM class-level indicators (exact matches)
FSM_CLASS_ATTRS = {
    "STATES",
    "TRANSITIONS",
    "STATE_MACHINE",
    "FSM",
    "ALLOWED_TRANSITIONS",
}

# FSM instance-level state tracking attributes
FSM_INSTANCE_ATTRS = {
    "_state",
    "_workflow_state",
    "_current_step",
    "_fsm_state",
    "_current_state",
}

# FSM method patterns
FSM_METHODS = {
    "transition",
    "can_transition",
    "apply_transition",
    "get_current_state",
    "set_state",
}


class OrchestratorFSMVisitor(ast.NodeVisitor):
    """AST visitor to detect FSM patterns in orchestrators.

    This visitor traverses Python AST to find FSM anti-patterns in
    orchestrator classes. Reducer classes are exempt from this check
    as they are meant to own state machines.

    Detection Patterns:
        1. Class-level: STATES, TRANSITIONS, STATE_* constants
        2. Instance-level: _state, _workflow_state, _current_step
        3. Method-level: transition(), can_transition(), get_current_state()

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
        self._in_orchestrator = False
        self._current_class_name: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track orchestrator class context, exempt reducers.

        Args:
            node: The ClassDef AST node being visited.
        """
        is_orchestrator = "Orchestrator" in node.name
        is_reducer = "Reducer" in node.name

        # Only check orchestrators, reducers are exempt
        old_in_orchestrator = self._in_orchestrator
        old_class_name = self._current_class_name

        self._in_orchestrator = is_orchestrator and not is_reducer
        self._current_class_name = node.name

        if self._in_orchestrator:
            self._check_class_level_fsm(node)

        self.generic_visit(node)

        self._in_orchestrator = old_in_orchestrator
        self._current_class_name = old_class_name

    def _check_class_level_fsm(self, node: ast.ClassDef) -> None:
        """Check for FSM class attributes.

        Detects patterns like:
            - STATES = [...]
            - TRANSITIONS = {...}
            - STATE_* = "..." (prefix pattern)
            - ALLOWED_TRANSITIONS = {...}

        Args:
            node: The ClassDef AST node being checked.
        """
        for item in node.body:
            # Check class-level assignments (STATES = [...], TRANSITIONS = {...})
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attr_name = target.id
                        # Check exact match FSM attributes
                        if attr_name in FSM_CLASS_ATTRS:
                            self._add_violation(
                                item,
                                f"Orchestrator has FSM class attribute '{attr_name}'",
                                "Move state machine logic to a Reducer. "
                                "Orchestrators are reaction planners.",
                            )
                        # Check STATE_* prefix pattern
                        elif attr_name.startswith("STATE_"):
                            self._add_violation(
                                item,
                                f"Orchestrator has FSM state constant '{attr_name}'",
                                "Move state constants to a Reducer. "
                                "Orchestrators should not define state values.",
                            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check for FSM methods in orchestrators.

        Detects methods like:
            - transition(), can_transition(), apply_transition()
            - get_current_state(), set_state()

        Args:
            node: The FunctionDef AST node being visited.
        """
        if self._in_orchestrator:
            if node.name in FSM_METHODS:
                self._add_violation(
                    node,
                    f"Orchestrator has FSM method '{node.name}()'",
                    "Move state machine logic to a Reducer. "
                    "Orchestrators should delegate state to reducers.",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for FSM instance attributes in orchestrators.

        Detects patterns like:
            - self._state = ...
            - self._workflow_state = ...
            - self._current_step = ...

        Args:
            node: The Assign AST node being visited.
        """
        if self._in_orchestrator:
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(
                    target.value, ast.Name
                ):
                    if target.value.id == "self" and target.attr in FSM_INSTANCE_ATTRS:
                        self._add_violation(
                            node,
                            f"Orchestrator has FSM instance attribute '{target.attr}'",
                            "Move state tracking to a Reducer. "
                            "Orchestrators should not own workflow state.",
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
            node: The AST node where the violation was found.
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
                target_type="orchestrator",
                target_name=self._current_class_name or "unknown",
                message=message,
                location=location,
                suggestion=suggestion,
            )
        )


def validate_no_orchestrator_fsm(file_path: str) -> ModelFileValidationResult:
    """Validate that orchestrators do not implement workflow FSMs.

    This function checks a Python file for FSM patterns in orchestrator code
    that violate ARCH-003. Orchestrators should delegate state management
    to reducers, not implement their own state machines.

    Note:
        Reducers are EXEMPT from this rule - they are meant to own state machines.

    Args:
        file_path: Path to the Python file to validate.

    Returns:
        ModelArchitectureValidationResult with:
            - valid=True if no FSM patterns found in orchestrators
            - valid=False with violations if FSM patterns detected

    Detection Patterns:
        1. Class-level FSM patterns:
           - STATES = [...], TRANSITIONS = {...}
           - STATE_* constants, ALLOWED_TRANSITIONS

        2. Instance-level state tracking:
           - self._state, self._current_step, self._workflow_state

        3. FSM method patterns:
           - can_transition(), apply_transition(), get_current_state()
           - transition(), set_state()

    Exclusions:
        - Reducer classes (they OWN state machines)
        - Non-orchestrator classes
        - Non-Python files
    """
    path = Path(file_path)

    # Skip non-existent or non-Python files
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

    visitor = OrchestratorFSMVisitor(file_path)
    visitor.visit(tree)

    return ModelFileValidationResult(
        valid=len(visitor.violations) == 0,
        violations=visitor.violations,
        files_checked=1,
        rules_checked=[RULE_ID],
    )


class RuleNoOrchestratorFSM(MixinFilePathRule):
    """Protocol-compliant rule: No workflow FSM in orchestrators.

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
            "Orchestrators must not implement workflow FSMs that duplicate reducer "
            "state transitions. Orchestrators are reaction planners, not state machine owners. "
            "Reducers may legitimately implement aggregate state machines."
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
        result = validate_no_orchestrator_fsm(file_path)

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
            message="Orchestrator FSM violation detected",
        )


__all__ = ["validate_no_orchestrator_fsm", "RuleNoOrchestratorFSM"]
