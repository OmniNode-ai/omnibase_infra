# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""AST-based Execution Shape Validator for ONEX 4-Node Architecture.

This module provides static analysis validation for ONEX handlers to ensure
compliance with execution shape constraints. It uses Python AST to detect
forbidden patterns in handler code without runtime execution.

Execution Shape Rules (from ONEX architecture):
    - EFFECT: Can return EVENT, COMMAND. Cannot return PROJECTION.
    - COMPUTE: Can return any message type. No restrictions.
    - REDUCER: Can return PROJECTION. Cannot return EVENT. Cannot access system time.
    - ORCHESTRATOR: Can return COMMAND, EVENT. Cannot return INTENT, PROJECTION.

All handlers are forbidden from direct publish operations (.publish(), .send_event(), .emit()).

Limitations:
    This validator uses AST-based static analysis, which has inherent limitations:

    **Detection Capabilities:**
    - Handler type detection via class name suffix (e.g., `OrderEffectHandler`)
    - Handler type detection via base class (e.g., `class X(EffectHandler)`)
    - Handler type detection via decorator (e.g., `@effect_handler`)
    - Return type analysis from annotations and return statements
    - Direct publish method call detection (`.publish()`, `.emit()`, etc.)
    - System time access detection (`time.time()`, `datetime.now()`, etc.)

    **Known Limitations:**
    - Cannot detect dynamically constructed type names or factory patterns
      (e.g., `type(f"{prefix}Event", (BaseEvent,), {})`)
    - Cannot follow imports to resolve types from other modules
      (e.g., `from events import OrderEvent` then `return OrderEvent(...)`)
    - Cannot detect indirect calls via reflection
      (e.g., `getattr(self, "publish")()`, `method_name = "emit"; getattr(self, method_name)()`)
    - Substring matching may produce false positives for non-message types
      (e.g., `EventProcessor`, `CommandLineArgs`, `IntentionallyBlank`)
    - Handler type detection requires conventional naming or explicit markers
    - Does not analyze runtime behavior, conditional paths, or dynamic dispatch
    - Cannot validate return types from external function calls
      (e.g., `return some_factory.create_event()`)

    **False Positive Scenarios:**
    - Variable names containing message category keywords (e.g., `event_count`)
    - Classes with category-like suffixes that aren't message types
    - Overridden methods that don't follow base class behavior
    - Helper classes or utilities with message-like names

    For runtime validation that handles dynamic cases, use
    :class:`RuntimeShapeValidator` which validates actual return values.

Performance:
    This module uses a module-level singleton validator (`_validator`) for
    efficiency. The `validate_execution_shapes()` and `validate_execution_shapes_ci()`
    functions use this cached instance. The singleton is thread-safe because:

    - The validator is stateless after initialization (only stores immutable rules)
    - AST parsing happens per-file with no shared mutable state
    - Each validation creates fresh ModelDetectedNodeInfo and violation objects

    For custom validation rules, create a new `ExecutionShapeValidator` instance.
    For repeated validation of the same files, the singleton pattern provides
    optimal performance.

Usage:
    >>> from omnibase_infra.validation.validator_execution_shape import (
    ...     validate_execution_shapes,
    ...     validate_execution_shapes_ci,
    ... )
    >>> violations = validate_execution_shapes(Path("src/handlers"))
    >>> result = validate_execution_shapes_ci(Path("src/handlers"))
    >>> if not result.passed:
    ...     print(f"Found {result.blocking_count} blocking violations")
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import (
    EnumExecutionShapeViolation,
    EnumNodeArchetype,
    EnumNodeOutputType,
    EnumValidationSeverity,
)
from omnibase_infra.models.validation.model_execution_shape_rule import (
    ModelExecutionShapeRule,
)
from omnibase_infra.models.validation.model_execution_shape_validation_result import (
    ModelExecutionShapeValidationResult,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)
from omnibase_infra.validation.mixin_execution_shape_violation_checks import (
    MixinExecutionShapeViolationChecks,
)
from omnibase_infra.validation.mixin_node_archetype_detection import (
    MixinNodeArchetypeDetection,
)

logger = logging.getLogger(__name__)

# Canonical execution shape rules for each node archetype.
# All output types use EnumNodeOutputType consistently (EVENT, COMMAND, INTENT, PROJECTION).
# EnumMessageCategory (EVENT, COMMAND, INTENT) is used for message routing topics,
# while EnumNodeOutputType is used for execution shape validation.
EXECUTION_SHAPE_RULES: dict[EnumNodeArchetype, ModelExecutionShapeRule] = {
    EnumNodeArchetype.EFFECT: ModelExecutionShapeRule(
        node_archetype=EnumNodeArchetype.EFFECT,
        allowed_return_types=[EnumNodeOutputType.EVENT, EnumNodeOutputType.COMMAND],
        forbidden_return_types=[EnumNodeOutputType.PROJECTION],
        can_publish_directly=False,
        can_access_system_time=True,
    ),
    EnumNodeArchetype.COMPUTE: ModelExecutionShapeRule(
        node_archetype=EnumNodeArchetype.COMPUTE,
        allowed_return_types=[
            EnumNodeOutputType.EVENT,
            EnumNodeOutputType.COMMAND,
            EnumNodeOutputType.INTENT,
            EnumNodeOutputType.PROJECTION,
        ],
        forbidden_return_types=[],
        can_publish_directly=False,
        can_access_system_time=True,
    ),
    EnumNodeArchetype.REDUCER: ModelExecutionShapeRule(
        node_archetype=EnumNodeArchetype.REDUCER,
        allowed_return_types=[EnumNodeOutputType.PROJECTION],
        forbidden_return_types=[EnumNodeOutputType.EVENT],
        can_publish_directly=False,
        can_access_system_time=False,
    ),
    EnumNodeArchetype.ORCHESTRATOR: ModelExecutionShapeRule(
        node_archetype=EnumNodeArchetype.ORCHESTRATOR,
        allowed_return_types=[EnumNodeOutputType.COMMAND, EnumNodeOutputType.EVENT],
        forbidden_return_types=[
            EnumNodeOutputType.INTENT,
            EnumNodeOutputType.PROJECTION,
        ],
        can_publish_directly=False,
        can_access_system_time=True,
    ),
}


class ModelDetectedNodeInfo(BaseModel):
    """Information about a detected node in source code during validation.

    This Pydantic model replaces the previous dataclass implementation
    to comply with ONEX requirements for Pydantic-based data structures.

    Attributes:
        name: The class or function name.
        node_archetype: The detected node archetype (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
        node: The AST node representing the detected element.
        line_number: The line number where the element is defined.
        file_path: The absolute path to the source file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    node_archetype: EnumNodeArchetype
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    line_number: int
    file_path: str


class ExecutionShapeValidator(
    MixinNodeArchetypeDetection,
    MixinExecutionShapeViolationChecks,
):
    """AST-based validator for ONEX handler execution shapes.

    This validator parses Python source files and analyzes handler classes/functions
    to detect violations of ONEX 4-node architecture constraints.

    Detection capabilities:
        - REDUCER_RETURNS_EVENTS: Reducer handler returns Event types
        - ORCHESTRATOR_RETURNS_INTENTS: Orchestrator returns Intent types
        - ORCHESTRATOR_RETURNS_PROJECTIONS: Orchestrator returns Projection types
        - EFFECT_RETURNS_PROJECTIONS: Effect handler returns Projection types
        - HANDLER_DIRECT_PUBLISH: Any handler calls .publish() directly
        - REDUCER_ACCESSES_SYSTEM_TIME: Reducer calls time.time(), datetime.now(), etc.

    Architecture:
        This class uses mixin-based composition to keep method count manageable:
        - MixinNodeArchetypeDetection: Handler and archetype detection from AST
        - MixinExecutionShapeViolationChecks: Violation checking and reporting

    Thread Safety:
        ExecutionShapeValidator instances are stateless after initialization.
        The rules dictionary is immutable and no per-validation state is stored.
        Multiple threads can safely call validate_file() or validate_directory()
        on the same validator instance concurrently.

        **Why no locks are needed** (unlike RoutingCoverageValidator):
        - No lazy initialization: Rules are assigned in __init__, not deferred
        - No mutable state: The rules dict reference never changes after __init__
        - No caching: Each validation creates fresh local variables (handlers, violations)
        - Module-level singleton created at import time (before any threads)

        This is fundamentally different from RoutingCoverageValidator which uses
        double-checked locking because it has lazy initialization with cached state.

    Performance:
        For repeated validation (e.g., CI pipelines), use the module-level
        functions `validate_execution_shapes()` or `validate_execution_shapes_ci()`
        which use a cached singleton validator for optimal performance.

        Creating new validator instances is cheap but unnecessary in most cases.
        Create a new instance only if you need custom rules or isolation.

    Example:
        >>> validator = ExecutionShapeValidator()
        >>> violations = validator.validate_file(Path("src/handlers/my_handler.py"))
        >>> for v in violations:
        ...     print(v.format_for_ci())
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self._rules = EXECUTION_SHAPE_RULES

    def validate_file(
        self, file_path: Path
    ) -> list[ModelExecutionShapeViolationResult]:
        """Validate a single Python file for execution shape violations.

        Args:
            file_path: Path to the Python file to validate.

        Returns:
            List of detected violations. Empty list if no violations found.
            For syntax errors, returns a single SYNTAX_ERROR violation.

        Note:
            Syntax errors are returned as violations rather than raised as exceptions.
            This enables CI pipelines to collect all violations across files rather
            than failing on the first unparseable file. The violation includes the
            syntax error details for debugging.
        """
        source = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            # Syntax error is a file-level issue, not a handler-specific violation.
            # Use SYNTAX_ERROR violation type for AST parse failures.
            # node_archetype is None because we can't analyze the code structure.
            logger.warning(
                "Syntax error in file",
                extra={"file": str(file_path), "error": str(e)},
            )
            return [
                ModelExecutionShapeViolationResult(
                    violation_type=EnumExecutionShapeViolation.SYNTAX_ERROR,
                    node_archetype=None,  # Cannot determine archetype from unparseable file
                    file_path=str(file_path.resolve()),
                    line_number=e.lineno or 1,
                    message=(
                        f"Validation error: Cannot parse Python source file. "
                        f"Syntax error at line {e.lineno or 1}: {e.msg}. "
                        f"File: {file_path.name}. Fix the syntax error to enable "
                        f"execution shape validation."
                    ),
                    severity=EnumValidationSeverity.ERROR,
                )
            ]

        violations: list[ModelExecutionShapeViolationResult] = []

        # Find all handlers in the file
        handlers = self._find_handlers(tree, str(file_path.resolve()))

        # Validate each handler
        for handler in handlers:
            handler_violations = self._validate_handler(handler)
            violations.extend(handler_violations)

        return violations

    def validate_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Validate all Python files in a directory.

        Args:
            directory: Path to the directory to validate.
            recursive: If True, recursively validate subdirectories.

        Returns:
            List of all detected violations across all files.
            Syntax errors in individual files are included as SYNTAX_ERROR violations
            rather than causing the directory validation to fail.

        Note:
            Files starting with underscore (e.g., __init__.py) are skipped.
            Non-Python files are ignored.
        """
        violations: list[ModelExecutionShapeViolationResult] = []
        pattern = "**/*.py" if recursive else "*.py"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and not file_path.name.startswith("_"):
                try:
                    file_violations = self.validate_file(file_path)
                    violations.extend(file_violations)
                except Exception as e:
                    # Unexpected errors (I/O errors, permission issues, etc.)
                    # Log and continue to validate remaining files
                    logger.warning(
                        "Failed to validate file",
                        extra={
                            "file": str(file_path),
                            "error_type": type(e).__name__,
                            "error": str(e),
                        },
                    )
                    continue

        return violations


def validate_execution_shapes(
    directory: Path,
) -> list[ModelExecutionShapeViolationResult]:
    """Validate all Python files in directory for execution shape violations.

    This is the main entry point for batch validation of handler files.

    Uses a cached singleton validator for performance in hot paths.

    Args:
        directory: Path to the directory containing handler files.

    Returns:
        List of all detected violations across all files.

    Example:
        >>> violations = validate_execution_shapes(Path("src/handlers"))
        >>> for v in violations:
        ...     print(f"{v.file_path}:{v.line_number}: {v.message}")
    """
    # Use cached singleton validator for performance
    # Singleton is safe because ExecutionShapeValidator is stateless after init
    return _validator.validate_directory(directory)


def validate_execution_shapes_ci(
    directory: Path,
) -> ModelExecutionShapeValidationResult:
    """CI gate for execution shape validation.

    This function is designed for CI pipeline integration. It returns a
    structured result model containing the pass/fail status and all violations
    for reporting.

    Args:
        directory: Path to the directory to validate.

    Returns:
        ModelExecutionShapeValidationResult containing:
            - passed: True if no error-severity violations found.
            - violations: Complete list of all violations for reporting.
            - Convenience methods for formatting and inspection.

    Example:
        >>> result = validate_execution_shapes_ci(Path("src/handlers"))
        >>> if not result.passed:
        ...     for line in result.format_for_ci():
        ...         print(line)
        ...     sys.exit(1)

    .. versionchanged:: 0.6.0
        Changed return type from ``tuple[bool, list[...]]`` to
        ``ModelExecutionShapeValidationResult`` (OMN-1003).
    """
    violations = validate_execution_shapes(directory)
    return ModelExecutionShapeValidationResult.from_violations(violations)


def get_execution_shape_rules() -> dict[EnumNodeArchetype, ModelExecutionShapeRule]:
    """Get the canonical execution shape rules.

    Returns a shallow copy of the rules dictionary to prevent modification
    of the module-level canonical rules. The values (ModelExecutionShapeRule
    instances) are frozen Pydantic models (immutable), so a shallow copy
    is sufficient for immutability - callers cannot modify either the
    dictionary structure or the rule objects.

    Returns:
        Dictionary mapping node archetypes to their execution shape rules.
        Callers may safely modify the returned dictionary without affecting
        the canonical rules.

    Example:
        >>> rules = get_execution_shape_rules()
        >>> # Safe to modify the returned dict (won't affect original)
        >>> del rules[EnumNodeArchetype.COMPUTE]
        >>>
        >>> # But rule objects are immutable (will raise TypeError)
        >>> rules[EnumNodeArchetype.EFFECT].can_publish_directly = True  # Raises!
    """
    return EXECUTION_SHAPE_RULES.copy()


# ==============================================================================
# Module-Level Singleton Validator
# ==============================================================================
#
# Performance Optimization: The ExecutionShapeValidator is stateless after
# initialization (only stores rules dictionary). Creating new instances on
# every validation call is wasteful in hot paths. Instead, we use a
# module-level singleton.
#
# Why a singleton is safe here:
# - The validator's rules dictionary is immutable after initialization
# - No per-validation state is stored in the validator instance
# - AST parsing happens per-file (no shared mutable state)
# - The ModelDetectedNodeInfo and violations are created fresh for each file
#
# Thread Safety:
# - The singleton is created at module import time (before any threads)
# - All read operations on the rules dictionary are thread-safe
# - Each validation creates fresh local state (handlers, violations lists)
# - No locks are needed because there's no shared mutable state
#
# Contrast with RoutingCoverageValidator:
#   RoutingCoverageValidator uses double-checked locking with threading.Lock
#   because it has LAZY initialization (discovery happens on first use) and
#   CACHED STATE (discovered types and routes are stored). The lock prevents
#   race conditions where two threads might both trigger discovery.
#
#   ExecutionShapeValidator needs no locks because:
#   1. Initialization is EAGER (rules assigned in __init__ at module load)
#   2. State is IMMUTABLE (rules dict never changes after construction)
#   3. No caching - each validate_file() call does fresh AST parsing
#
#   This is a key architectural distinction: stateless validators can use
#   simple singletons, while stateful validators need synchronization.
#
# When NOT to use the singleton:
# - If you need custom execution shape rules (create your own instance)
# - If you need to mock the validator in tests (inject or patch)
# - If you're validating in a context that requires isolation

_validator = ExecutionShapeValidator()


__all__ = [
    "EXECUTION_SHAPE_RULES",
    "ExecutionShapeValidator",
    "ModelDetectedNodeInfo",
    "ModelExecutionShapeValidationResult",
    "get_execution_shape_rules",
    "validate_execution_shapes",
    "validate_execution_shapes_ci",
]
