# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""AST-based Declarative Node Validator for ONEX.

Contract-driven validator to enforce the ONEX declarative node policy:

- Node classes MUST only extend base classes without custom logic
- Only __init__ with super().__init__(container) is allowed
- No custom methods, properties, or instance variables
- All behavior should be defined in contract.yaml

The validator uses Python AST to detect forbidden patterns without runtime execution.

Exemption Mechanism:
    ``# ONEX_EXCLUDE: declarative_node`` comment on the class line exempts that class.

Usage:
    Programmatic::

        >>> from pathlib import Path
        >>> from omnibase_infra.validation.validator_declarative_node import (
        ...     ValidatorDeclarativeNode,
        ... )
        >>> validator = ValidatorDeclarativeNode()
        >>> result = validator.validate(Path("src/nodes"))
        >>> if not result.is_valid:
        ...     for issue in result.issues:
        ...         print(f"{issue.file_path}:{issue.line_number}: {issue.message}")

    Module-level convenience functions::

        >>> from omnibase_infra.validation.validator_declarative_node import (
        ...     validate_declarative_nodes,
        ...     validate_declarative_nodes_ci,
        ... )
        >>> violations = validate_declarative_nodes(Path("src/nodes"))
        >>> result = validate_declarative_nodes_ci(Path("src/nodes"))
        >>> if not result.passed:
        ...     print(f"Found {result.blocking_count} imperative nodes")

    CLI::

        python -m omnibase_infra.validation.validator_declarative_node src/nodes

See Also:
    - CLAUDE.md: MANDATORY: Declarative Nodes section
    - ValidatorBase: Base class for contract-driven validators

Limitations:
    - **Shallow inheritance detection**: Only direct base classes (NodeEffect,
      NodeCompute, NodeReducer, NodeOrchestrator) are recognized. Indirect
      inheritance through custom intermediate classes is not detected.
    - **Exemption comment window**: The ONEX_EXCLUDE comment must appear within
      3 lines before the class definition. Classes with many decorators may
      need the comment placed closer to the class definition.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from typing import ClassVar

from omnibase_core.models.common.model_validation_issue import ModelValidationIssue
from omnibase_core.models.contracts.subcontracts.model_validator_subcontract import (
    ModelValidatorSubcontract,
)
from omnibase_core.validation.validator_base import ValidatorBase
from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.enums.enum_declarative_node_violation import (
    EnumDeclarativeNodeViolation,
)
from omnibase_infra.models.validation.model_declarative_node_validation_result import (
    ModelDeclarativeNodeValidationResult,
)
from omnibase_infra.models.validation.model_declarative_node_violation import (
    ModelDeclarativeNodeViolation,
)

logger = logging.getLogger(__name__)

# Node base class names that indicate a class is an ONEX node
_NODE_BASE_CLASSES: frozenset[str] = frozenset(
    {
        "NodeEffect",
        "NodeCompute",
        "NodeReducer",
        "NodeOrchestrator",
    }
)

# Exemption comment pattern
_ONEX_EXCLUDE_PATTERN = "ONEX_EXCLUDE:"
_ONEX_EXCLUDE_DECLARATIVE = "declarative_node"

# Maximum file size to process (in bytes)
_MAX_FILE_SIZE_BYTES: int = 500_000  # 500KB


def _get_source_line(source_lines: list[str], line_number: int) -> str:
    """Get a source line safely (1-indexed).

    Args:
        source_lines: List of source lines.
        line_number: 1-indexed line number.

    Returns:
        The source line stripped, or empty string if out of bounds.
    """
    if 0 < line_number <= len(source_lines):
        return source_lines[line_number - 1].strip()
    return ""


def _is_class_exempted(source_lines: list[str], class_line: int) -> bool:
    """Check if a class is exempted via ONEX_EXCLUDE comment.

    Checks the class line and the 3 lines before it for exemption comment.
    This covers common patterns like:
    - Comment on same line as class
    - Comment on line immediately before class
    - Comment before decorators (up to 2 decorators)

    Args:
        source_lines: List of source lines.
        class_line: 1-indexed line number of the class definition.

    Returns:
        True if the class is exempted.

    Note:
        The 3-line lookback window may be insufficient for classes with:
        - More than 2 decorators above the exemption comment
        - Long multi-line decorators
        In such cases, place the exemption comment closer to the class
        definition or on the same line.
    """
    start_line = max(1, class_line - 3)
    for line_num in range(start_line, class_line + 1):
        line = _get_source_line(source_lines, line_num)
        if _ONEX_EXCLUDE_PATTERN in line and _ONEX_EXCLUDE_DECLARATIVE in line:
            return True
    return False


def _get_base_class_names(class_node: ast.ClassDef) -> set[str]:
    """Extract base class names from a class definition.

    Args:
        class_node: AST ClassDef node.

    Returns:
        Set of base class names (handles Name, Attribute, and Subscript nodes).

    Note:
        Subscript nodes represent generic types like NodeReducer["State", "Output"].
        We extract the base class name from the subscript value.
    """
    base_names: set[str] = set()
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            base_names.add(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.add(base.attr)
        elif isinstance(base, ast.Subscript):
            # Handle generic types like NodeReducer["State", "Output"]
            subscript_value = base.value
            if isinstance(subscript_value, ast.Name):
                base_names.add(subscript_value.id)
            elif isinstance(subscript_value, ast.Attribute):
                base_names.add(subscript_value.attr)
    return base_names


def _is_node_class(class_node: ast.ClassDef) -> bool:
    """Check if a class is an ONEX node class.

    Args:
        class_node: AST ClassDef node.

    Returns:
        True if the class inherits from a known node base class.

    Note:
        This function only checks direct base class names. Indirect inheritance
        (e.g., class MyNode(MyCustomBase) where MyCustomBase extends NodeEffect)
        is NOT detected. This is intentional per ONEX policy which discourages
        deep inheritance hierarchies in node classes.
    """
    base_names = _get_base_class_names(class_node)
    return bool(base_names & _NODE_BASE_CLASSES)


def _is_valid_init(func_node: ast.FunctionDef, source_lines: list[str]) -> bool:
    """Check if __init__ method is valid (only super().__init__ call).

    A valid __init__:
    - May have a docstring (Expr with Constant str)
    - Must have exactly one other statement: super().__init__(container)
    - May have pass statement if also has super() call

    Args:
        func_node: AST FunctionDef node for __init__.
        source_lines: Source lines for context.

    Returns:
        True if the __init__ is valid (declarative).
    """
    body = func_node.body

    # Filter out docstrings and pass statements
    significant_stmts = []
    for stmt in body:
        # Skip docstrings (Expr containing a Constant string)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if isinstance(stmt.value.value, str):
                continue
        # Skip pass statements
        if isinstance(stmt, ast.Pass):
            continue
        significant_stmts.append(stmt)

    # Should have exactly one significant statement
    if len(significant_stmts) != 1:
        return False

    stmt = significant_stmts[0]

    # Must be an Expr containing a Call
    if not isinstance(stmt, ast.Expr):
        return False
    if not isinstance(stmt.value, ast.Call):
        return False

    call = stmt.value

    # Check if it's super().__init__(...)
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "__init__":
        return False
    if not isinstance(call.func.value, ast.Call):
        return False

    super_call = call.func.value
    if not isinstance(super_call.func, ast.Name):
        return False
    if super_call.func.id != "super":
        return False

    # Validate super() has no args/keywords (must be bare super())
    if super_call.args or super_call.keywords:
        return False

    # Validate __init__ call has exactly 1 positional arg named "container"
    if len(call.args) != 1:
        return False
    if call.keywords:
        return False
    if not isinstance(call.args[0], ast.Name):
        return False
    if call.args[0].id != "container":
        return False

    return True


def _find_instance_variables(func_node: ast.FunctionDef) -> list[tuple[int, str]]:
    """Find instance variable assignments in __init__ (excluding docstrings).

    Uses ast.walk() to recursively find all assignments including those nested
    in if/for/try/while blocks. Handles tuple/list unpacking targets.

    Args:
        func_node: AST FunctionDef node for __init__.

    Returns:
        List of (line_number, variable_name) tuples for instance variables.
    """
    instance_vars: list[tuple[int, str]] = []

    def _extract_self_attrs(target: ast.AST, lineno: int) -> None:
        """Extract self.x attributes from a target, handling tuple/list unpacking."""
        if isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                instance_vars.append((lineno, target.attr))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                _extract_self_attrs(elt, lineno)

    for node in ast.walk(func_node):
        # Skip the docstring at the start of __init__
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node in func_node.body
        ):
            continue

        # Check for self.xxx = ... assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _extract_self_attrs(target, node.lineno)
        elif isinstance(node, ast.AnnAssign):
            if node.target is not None:
                _extract_self_attrs(node.target, node.lineno)
        elif isinstance(node, ast.AugAssign):
            # Handle self.x += ... augmented assignments
            _extract_self_attrs(node.target, node.lineno)

    return instance_vars


def _validate_node_class(
    class_node: ast.ClassDef,
    file_path: Path,
    source_lines: list[str],
) -> list[ModelDeclarativeNodeViolation]:
    """Validate a single node class for declarative compliance.

    Args:
        class_node: AST ClassDef node.
        file_path: Path to the source file.
        source_lines: Source lines for context.

    Returns:
        List of violations found in this class.
    """
    violations: list[ModelDeclarativeNodeViolation] = []
    class_name = class_node.name

    # Check for class-level variables (excluding type annotations without values)
    for stmt in class_node.body:
        # Skip docstrings
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if isinstance(stmt.value.value, str):
                continue

        # Skip pass statements
        if isinstance(stmt, ast.Pass):
            continue

        # Skip function definitions (handled separately)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Class variable assignment
        if isinstance(stmt, ast.Assign):
            snippet = _get_source_line(source_lines, stmt.lineno)
            violations.append(
                ModelDeclarativeNodeViolation(
                    file_path=file_path,
                    line_number=stmt.lineno,
                    violation_type=EnumDeclarativeNodeViolation.CLASS_VARIABLE,
                    code_snippet=snippet,
                    suggestion=EnumDeclarativeNodeViolation.CLASS_VARIABLE.suggestion,
                    node_class_name=class_name,
                )
            )

        # Annotated assignment with value
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            snippet = _get_source_line(source_lines, stmt.lineno)
            violations.append(
                ModelDeclarativeNodeViolation(
                    file_path=file_path,
                    line_number=stmt.lineno,
                    violation_type=EnumDeclarativeNodeViolation.CLASS_VARIABLE,
                    code_snippet=snippet,
                    suggestion=EnumDeclarativeNodeViolation.CLASS_VARIABLE.suggestion,
                    node_class_name=class_name,
                )
            )

    # Check methods
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef):
            method_name = item.name

            # Check for properties (including qualified decorators like functools.cached_property)
            for decorator in item.decorator_list:
                is_property = False
                if (
                    isinstance(decorator, ast.Name)
                    and decorator.id
                    in {
                        "property",
                        "cached_property",
                    }
                ) or (
                    isinstance(decorator, ast.Attribute)
                    and decorator.attr
                    in {
                        "property",
                        "cached_property",
                    }
                ):
                    is_property = True

                if is_property:
                    snippet = _get_source_line(source_lines, item.lineno)
                    violations.append(
                        ModelDeclarativeNodeViolation(
                            file_path=file_path,
                            line_number=item.lineno,
                            violation_type=EnumDeclarativeNodeViolation.CUSTOM_PROPERTY,
                            code_snippet=snippet,
                            suggestion=EnumDeclarativeNodeViolation.CUSTOM_PROPERTY.suggestion,
                            node_class_name=class_name,
                            method_name=method_name,
                        )
                    )
                    break
            else:
                # Not a property, check if it's a valid method
                if method_name == "__init__":
                    # Validate __init__
                    if not _is_valid_init(item, source_lines):
                        snippet = _get_source_line(source_lines, item.lineno)
                        violations.append(
                            ModelDeclarativeNodeViolation(
                                file_path=file_path,
                                line_number=item.lineno,
                                violation_type=EnumDeclarativeNodeViolation.INIT_CUSTOM_LOGIC,
                                code_snippet=snippet,
                                suggestion=EnumDeclarativeNodeViolation.INIT_CUSTOM_LOGIC.suggestion,
                                node_class_name=class_name,
                                method_name="__init__",
                            )
                        )

                    # Check for instance variables
                    instance_vars = _find_instance_variables(item)
                    for line_num, var_name in instance_vars:
                        snippet = _get_source_line(source_lines, line_num)
                        violations.append(
                            ModelDeclarativeNodeViolation(
                                file_path=file_path,
                                line_number=line_num,
                                violation_type=EnumDeclarativeNodeViolation.INSTANCE_VARIABLE,
                                code_snippet=snippet,
                                suggestion=EnumDeclarativeNodeViolation.INSTANCE_VARIABLE.suggestion,
                                node_class_name=class_name,
                                method_name=f"self.{var_name}",
                            )
                        )
                else:
                    # Custom method (not __init__)
                    snippet = _get_source_line(source_lines, item.lineno)
                    violations.append(
                        ModelDeclarativeNodeViolation(
                            file_path=file_path,
                            line_number=item.lineno,
                            violation_type=EnumDeclarativeNodeViolation.CUSTOM_METHOD,
                            code_snippet=snippet,
                            suggestion=EnumDeclarativeNodeViolation.CUSTOM_METHOD.suggestion,
                            node_class_name=class_name,
                            method_name=method_name,
                        )
                    )

        # Check for async methods
        elif isinstance(item, ast.AsyncFunctionDef):
            snippet = _get_source_line(source_lines, item.lineno)
            violations.append(
                ModelDeclarativeNodeViolation(
                    file_path=file_path,
                    line_number=item.lineno,
                    violation_type=EnumDeclarativeNodeViolation.CUSTOM_METHOD,
                    code_snippet=snippet,
                    suggestion=EnumDeclarativeNodeViolation.CUSTOM_METHOD.suggestion,
                    node_class_name=class_name,
                    method_name=item.name,
                )
            )

    return violations


# Map EnumDeclarativeNodeViolation to contract rule IDs
_VIOLATION_TO_RULE_ID: dict[EnumDeclarativeNodeViolation, str] = {
    EnumDeclarativeNodeViolation.CUSTOM_METHOD: "DECL-001",
    EnumDeclarativeNodeViolation.CUSTOM_PROPERTY: "DECL-002",
    EnumDeclarativeNodeViolation.INIT_CUSTOM_LOGIC: "DECL-003",
    EnumDeclarativeNodeViolation.INSTANCE_VARIABLE: "DECL-004",
    EnumDeclarativeNodeViolation.CLASS_VARIABLE: "DECL-005",
    EnumDeclarativeNodeViolation.SYNTAX_ERROR: "DECL-006",
    EnumDeclarativeNodeViolation.NO_NODE_CLASS: "DECL-007",
}


# =============================================================================
# SHARED CORE VALIDATION FUNCTIONS
# =============================================================================
# These internal functions contain the core validation logic that is shared
# between the ValidatorDeclarativeNode class and the module-level functions.
# This eliminates duplication while providing multiple API styles.
# =============================================================================


def _validate_file_core(
    file_path: Path,
) -> list[ModelDeclarativeNodeViolation]:
    """Core file validation logic shared by class and legacy functions.

    Performs the following validation steps:
    1. Check file size against limit
    2. Read file content
    3. Parse AST
    4. Find node classes and validate for declarative compliance

    Args:
        file_path: Path to the node.py file to validate.

    Returns:
        List of ModelDeclarativeNodeViolation instances for violations found.
        Returns empty list if file cannot be read/parsed or is too large.
    """
    # Check file size
    try:
        file_size = file_path.stat().st_size
        if file_size > _MAX_FILE_SIZE_BYTES:
            logger.warning(
                "Skipping file %s: size %d exceeds limit %d",
                file_path,
                file_size,
                _MAX_FILE_SIZE_BYTES,
            )
            return []
    except OSError as e:
        logger.warning("Cannot stat file %s: %s", file_path, e)
        return []

    # Read file
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Cannot read file %s: %s", file_path, e)
        return []

    source_lines = source.splitlines()

    # Parse AST
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        return [
            ModelDeclarativeNodeViolation(
                file_path=file_path,
                line_number=e.lineno or 1,
                violation_type=EnumDeclarativeNodeViolation.SYNTAX_ERROR,
                code_snippet=str(e.msg) if e.msg else "Syntax error",
                suggestion=EnumDeclarativeNodeViolation.SYNTAX_ERROR.suggestion,
                severity=EnumValidationSeverity.ERROR,
            )
        ]

    violations: list[ModelDeclarativeNodeViolation] = []
    found_node_class = False

    # Find and validate node classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if _is_node_class(node):
                found_node_class = True

                # Check for exemption
                if _is_class_exempted(source_lines, node.lineno):
                    logger.debug(
                        "Skipping exempted class %s in %s",
                        node.name,
                        file_path,
                    )
                    continue

                class_violations = _validate_node_class(node, file_path, source_lines)
                violations.extend(class_violations)

    # Emit warning for node.py files in nodes/ that don't contain node classes
    if not found_node_class and "nodes" in file_path.parts:
        violations.append(
            ModelDeclarativeNodeViolation(
                file_path=file_path,
                line_number=1,
                violation_type=EnumDeclarativeNodeViolation.NO_NODE_CLASS,
                code_snippet="# No Node class found in file",
                suggestion=EnumDeclarativeNodeViolation.NO_NODE_CLASS.suggestion,
                severity=EnumValidationSeverity.WARNING,
            )
        )

    return violations


def _validate_directory_with_count(
    directory: Path,
    recursive: bool = True,
) -> tuple[list[ModelDeclarativeNodeViolation], int]:
    """Validate all node.py files in a directory with single-pass traversal.

    This function performs a single directory traversal, collecting both
    violations and file count simultaneously for efficiency.

    Args:
        directory: Directory to scan for node.py files.
        recursive: If True, scan subdirectories.

    Returns:
        Tuple of (list of all violations found, count of files checked).
    """
    violations: list[ModelDeclarativeNodeViolation] = []
    files_checked = 0
    pattern = "**/node.py" if recursive else "node.py"

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            files_checked += 1
            file_violations = _validate_file_core(file_path)
            violations.extend(file_violations)

    return violations, files_checked


class ValidatorDeclarativeNode(ValidatorBase):
    """Contract-driven declarative node validator for ONEX.

    This validator uses AST analysis to detect imperative patterns in node.py files:
    - Custom methods beyond __init__ (should be in contract/handlers)
    - @property decorators (state should be in models)
    - Custom logic in __init__ beyond super().__init__(container)
    - Instance variable assignments (state belongs in container/models)
    - Class-level variable assignments (configuration belongs in contract)

    The validator is contract-driven via declarative_node.validation.yaml, supporting:
    - Configurable rules with enable/disable per rule
    - Per-rule severity overrides
    - Suppression comments for intentional exceptions
    - Glob-based file targeting and exclusion

    Thread Safety:
        ValidatorDeclarativeNode instances are NOT thread-safe due to internal mutable
        state inherited from ValidatorBase. When using parallel execution
        (e.g., pytest-xdist), create separate validator instances per worker.

    Attributes:
        validator_id: Unique identifier for this validator ("declarative_node").

    Usage Example:
        >>> from pathlib import Path
        >>> from omnibase_infra.validation.validator_declarative_node import (
        ...     ValidatorDeclarativeNode,
        ... )
        >>> validator = ValidatorDeclarativeNode()
        >>> result = validator.validate(Path("src/nodes"))
        >>> print(f"Valid: {result.is_valid}, Issues: {len(result.issues)}")

    CLI Usage:
        python -m omnibase_infra.validation.validator_declarative_node src/nodes
    """

    # ONEX_EXCLUDE: string_id - human-readable validator identifier
    validator_id: ClassVar[str] = "declarative_node"

    def _validate_file(
        self,
        path: Path,
        contract: ModelValidatorSubcontract,
    ) -> tuple[ModelValidationIssue, ...]:
        """Validate a single node.py file for declarative compliance.

        Uses AST analysis to detect imperative patterns:
        - Custom methods (not __init__)
        - Properties
        - Custom init logic
        - Instance variables
        - Class variables

        This method delegates to _validate_file_core for the actual validation,
        then converts ModelDeclarativeNodeViolation to ModelValidationIssue
        with contract-based rule filtering.

        Args:
            path: Path to the node.py file to validate.
            contract: Validator contract with rule configurations.

        Returns:
            Tuple of ModelValidationIssue instances for violations found.
        """
        # Use shared core validation logic
        violations = _validate_file_core(path)

        # Convert violations to ModelValidationIssue with contract filtering
        issues: list[ModelValidationIssue] = []
        for violation in violations:
            # Map violation type to rule ID
            rule_id = _VIOLATION_TO_RULE_ID.get(violation.violation_type)
            if rule_id is None:
                logger.warning(
                    "Unknown violation type %s, skipping",
                    violation.violation_type,
                )
                continue

            # Check if rule is enabled and get severity
            enabled, severity = self._get_rule_config(rule_id, contract)
            if not enabled:
                logger.debug(
                    "Rule %s is disabled, skipping violation",
                    rule_id,
                )
                continue

            # Convert to ModelValidationIssue
            context: dict[str, str] = {
                "violation_type": violation.violation_type.value,
            }
            if violation.node_class_name:
                context["class_name"] = violation.node_class_name
            if violation.method_name:
                context["method_name"] = violation.method_name

            issues.append(
                ModelValidationIssue(
                    severity=severity,
                    message=self._format_message(violation),
                    code=rule_id,
                    file_path=path,
                    line_number=violation.line_number,
                    rule_name=violation.violation_type.value,
                    suggestion=violation.suggestion,
                    context=context,
                )
            )

        return tuple(issues)

    def _format_message(self, violation: ModelDeclarativeNodeViolation) -> str:
        """Format a human-readable message for a violation.

        Args:
            violation: The violation to format.

        Returns:
            Human-readable message describing the violation.
        """
        vtype = violation.violation_type
        class_name = violation.node_class_name or "Unknown"

        if vtype == EnumDeclarativeNodeViolation.CUSTOM_METHOD:
            method = violation.method_name or "unknown"
            return f"Class '{class_name}' has custom method '{method}' - declarative nodes must not have custom methods"
        elif vtype == EnumDeclarativeNodeViolation.CUSTOM_PROPERTY:
            prop = violation.method_name or "unknown"
            return f"Class '{class_name}' has property '{prop}' - declarative nodes must not have properties"
        elif vtype == EnumDeclarativeNodeViolation.INIT_CUSTOM_LOGIC:
            return f"Class '{class_name}' has custom logic in __init__ - only super().__init__(container) is allowed"
        elif vtype == EnumDeclarativeNodeViolation.INSTANCE_VARIABLE:
            var = violation.method_name or "unknown"
            return f"Class '{class_name}' creates instance variable '{var}' - declarative nodes must not store state"
        elif vtype == EnumDeclarativeNodeViolation.CLASS_VARIABLE:
            return f"Class '{class_name}' has class variable - configuration should be in contract.yaml"
        elif vtype == EnumDeclarativeNodeViolation.SYNTAX_ERROR:
            return f"File has syntax error: {violation.code_snippet}"
        elif vtype == EnumDeclarativeNodeViolation.NO_NODE_CLASS:
            path = violation.file_path
            return f"File '{path.name}' is named node.py but contains no Node class"
        else:
            return f"Class '{class_name}' violates declarative node policy: {violation.code_snippet}"


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================
# These functions provide a simpler API for common use cases.
# They delegate to the shared core functions defined above.
# =============================================================================


def validate_declarative_node_in_file(
    file_path: Path,
) -> list[ModelDeclarativeNodeViolation]:
    """Validate a single node.py file for declarative compliance.

    Convenience function that delegates to the core validation logic.

    Args:
        file_path: Path to the node.py file.

    Returns:
        List of violations found (empty if compliant or not a node file).
    """
    return _validate_file_core(file_path)


def validate_declarative_nodes(
    directory: Path,
    recursive: bool = True,
) -> list[ModelDeclarativeNodeViolation]:
    """Validate all node.py files in a directory.

    Convenience function that validates all node.py files, returning only violations.
    For CI integration with file counts, use validate_declarative_nodes_ci instead.

    Args:
        directory: Directory to scan for node.py files.
        recursive: If True, scan subdirectories.

    Returns:
        List of all violations found.
    """
    violations, _ = _validate_directory_with_count(directory, recursive)
    return violations


def validate_declarative_nodes_ci(
    directory: Path,
    recursive: bool = True,
) -> ModelDeclarativeNodeValidationResult:
    """CI gate entry point for declarative node validation.

    Uses single-pass directory traversal for efficiency, collecting both
    violations and file count simultaneously.

    Args:
        directory: Directory to validate.
        recursive: If True, scan subdirectories.

    Returns:
        Result model with pass/fail status suitable for CI integration.
    """
    violations, files_checked = _validate_directory_with_count(directory, recursive)
    return ModelDeclarativeNodeValidationResult.from_violations(
        violations, files_checked
    )


__all__ = [
    "ValidatorDeclarativeNode",
    "validate_declarative_nodes",
    "validate_declarative_nodes_ci",
    "validate_declarative_node_in_file",
]


# CLI entry point
if __name__ == "__main__":
    sys.exit(ValidatorDeclarativeNode.main())
