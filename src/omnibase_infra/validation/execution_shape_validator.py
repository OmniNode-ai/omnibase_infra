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
    >>> from omnibase_infra.validation.execution_shape_validator import (
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
    EnumMessageCategory,
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

logger = logging.getLogger(__name__)

# Node archetype detection patterns
_NODE_ARCHETYPE_SUFFIX_MAP: dict[str, EnumNodeArchetype] = {
    "EffectHandler": EnumNodeArchetype.EFFECT,
    "ReducerHandler": EnumNodeArchetype.REDUCER,
    "OrchestratorHandler": EnumNodeArchetype.ORCHESTRATOR,
    "ComputeHandler": EnumNodeArchetype.COMPUTE,
    # Base class patterns
    "Effect": EnumNodeArchetype.EFFECT,
    "Reducer": EnumNodeArchetype.REDUCER,
    "Orchestrator": EnumNodeArchetype.ORCHESTRATOR,
    "Compute": EnumNodeArchetype.COMPUTE,
}

# Patterns for detecting message category types in return annotations.
#
# PATTERN MATCHING ORDER: More specific patterns (longer matches) are listed first
# within each category to prevent false positives. For example, "ModelEventMessage"
# should match "EventMessage" before "Event" to avoid partial matches.
#
# The patterns are checked using substring matching (`in` operator) in the order
# defined below. The dictionary preserves insertion order (Python 3.7+), so:
# 1. Compound patterns (e.g., "ModelEventMessage") - most specific
# 2. Prefix patterns (e.g., "ModelEvent", "EventMessage") - medium specificity
# 3. Simple patterns (e.g., "Event", "EVENT") - least specific, checked last
#
# NOTE: PROJECTION patterns use EnumNodeOutputType because PROJECTION is a node
# output type, not a message category for routing. Projections are produced by
# REDUCER nodes but are not routed via Kafka topics like EVENTs/COMMANDs/INTENTs.
#
# KNOWN LIMITATIONS:
# - Substring matching may produce false positives for non-message types that
#   happen to contain these keywords (e.g., "PreventEventLoop", "CommandLineArgs").
# - The fallback case-insensitive check in _detect_message_category() has the same
#   limitation but is intentionally lenient to catch non-standard naming conventions.
# - For strict validation, use the ONEX naming conventions (Model* prefix).
_MESSAGE_CATEGORY_PATTERNS: dict[str, EnumMessageCategory] = {
    # Event patterns - ordered by specificity (most specific first)
    "ModelEventMessage": EnumMessageCategory.EVENT,
    "EventMessage": EnumMessageCategory.EVENT,
    "ModelEvent": EnumMessageCategory.EVENT,
    "EVENT": EnumMessageCategory.EVENT,
    "Event": EnumMessageCategory.EVENT,
    # Command patterns - ordered by specificity (most specific first)
    "ModelCommandMessage": EnumMessageCategory.COMMAND,
    "CommandMessage": EnumMessageCategory.COMMAND,
    "ModelCommand": EnumMessageCategory.COMMAND,
    "COMMAND": EnumMessageCategory.COMMAND,
    "Command": EnumMessageCategory.COMMAND,
    # Intent patterns - ordered by specificity (most specific first)
    "ModelIntentMessage": EnumMessageCategory.INTENT,
    "IntentMessage": EnumMessageCategory.INTENT,
    "ModelIntent": EnumMessageCategory.INTENT,
    "INTENT": EnumMessageCategory.INTENT,
    "Intent": EnumMessageCategory.INTENT,
}

# Forbidden direct publish method names
_FORBIDDEN_PUBLISH_METHODS: frozenset[str] = frozenset(
    {
        "publish",
        "send_event",
        "emit",
        "emit_event",
        "dispatch",
        "dispatch_event",
    }
)

# System time access patterns (non-deterministic for reducers)
_SYSTEM_TIME_PATTERNS: frozenset[str] = frozenset(
    {
        "time.time",
        "datetime.now",
        "datetime.utcnow",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        # Django timezone support
        "timezone.now",
        "django.utils.timezone.now",
    }
)

# Module-level time function patterns
_TIME_FUNCTION_NAMES: frozenset[str] = frozenset(
    {
        "time",
        "now",
        "utcnow",
    }
)

# Mapping from EnumMessageCategory to EnumNodeOutputType for type normalization.
#
# Used by _is_return_type_allowed() to convert message categories to their
# corresponding node output types during execution shape validation. The rules
# use EnumNodeOutputType consistently, but message category detection may return
# EnumMessageCategory values that need conversion.
#
# FORWARD COMPATIBILITY NOTE:
# If new values are added to EnumMessageCategory in the future, this mapping
# must be updated. The .get() usage in _is_return_type_allowed() ensures
# graceful handling - unknown categories are disallowed by default rather
# than causing a runtime error.
#
# Both enums share EVENT, COMMAND, INTENT with same string values, so this
# mapping provides a clean bridge between the two type systems.
_MESSAGE_CATEGORY_TO_OUTPUT_TYPE: dict[EnumMessageCategory, EnumNodeOutputType] = {
    EnumMessageCategory.EVENT: EnumNodeOutputType.EVENT,
    EnumMessageCategory.COMMAND: EnumNodeOutputType.COMMAND,
    EnumMessageCategory.INTENT: EnumNodeOutputType.INTENT,
}

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


class ExecutionShapeValidator:
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

    def _find_handlers(
        self, tree: ast.AST, file_path: str
    ) -> list[ModelDetectedNodeInfo]:
        """Find all handler classes and functions in an AST.

        Detection methods:
            1. Class name suffix: *EffectHandler, *ReducerHandler, etc.
            2. Base class: class MyHandler(EffectHandler)
            3. Decorator: @node_archetype(EnumNodeArchetype.EFFECT)

        Args:
            tree: The parsed AST.
            file_path: Path to the source file.

        Returns:
            List of detected handlers with their archetype information.
        """
        handlers: list[ModelDetectedNodeInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                archetype = self._detect_node_archetype_from_class(node)
                if archetype is not None:
                    handlers.append(
                        ModelDetectedNodeInfo(
                            name=node.name,
                            node_archetype=archetype,
                            node=node,
                            line_number=node.lineno,
                            file_path=file_path,
                        )
                    )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                archetype = self._detect_node_archetype_from_decorator(node)
                if archetype is not None:
                    handlers.append(
                        ModelDetectedNodeInfo(
                            name=node.name,
                            node_archetype=archetype,
                            node=node,
                            line_number=node.lineno,
                            file_path=file_path,
                        )
                    )

        return handlers

    def _detect_node_archetype_from_class(
        self,
        node: ast.ClassDef,
    ) -> EnumNodeArchetype | None:
        """Detect node archetype from class definition.

        Checks:
            1. Class name suffix (e.g., OrderEffectHandler)
            2. Base class names (e.g., class OrderHandler(EffectHandler))
            3. Decorator on class

        Args:
            node: The class definition AST node.

        Returns:
            The detected node archetype, or None if not a handler.
        """
        # Check class name suffix
        for suffix, archetype in _NODE_ARCHETYPE_SUFFIX_MAP.items():
            if node.name.endswith(suffix):
                return archetype

        # Check base classes
        for base in node.bases:
            base_name = self._get_name_from_expr(base)
            if base_name is not None:
                for pattern, archetype in _NODE_ARCHETYPE_SUFFIX_MAP.items():
                    if base_name.endswith(pattern):
                        return archetype

        # Check decorators
        return self._detect_node_archetype_from_decorator(node)

    def _detect_node_archetype_from_decorator(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> EnumNodeArchetype | None:
        """Detect node archetype from decorators.

        Looks for patterns like:
            @node_archetype(EnumNodeArchetype.EFFECT)
            @effect_handler
            @reducer_handler

        Args:
            node: The AST node with decorators.

        Returns:
            The detected node archetype, or None if not found.
        """
        for decorator in node.decorator_list:
            # Check @node_archetype(EnumNodeArchetype.X) pattern
            if isinstance(decorator, ast.Call):
                func = decorator.func
                func_name = self._get_name_from_expr(func)
                if func_name in ("node_archetype", "handler_type") and decorator.args:
                    arg = decorator.args[0]
                    arg_str = self._get_name_from_expr(arg)
                    if arg_str is not None:
                        # Handle EnumNodeArchetype.EFFECT or just EFFECT
                        # Use exact matching to avoid false positives from substring matches
                        # (e.g., "side_effect" should not match "effect")
                        arg_upper = arg_str.upper()
                        for archetype in EnumNodeArchetype:
                            # Match exact enum member name: "EFFECT", "COMPUTE", etc.
                            if arg_upper == archetype.name:
                                return archetype
                            # Match qualified name: "EnumNodeArchetype.EFFECT"
                            if arg_upper.endswith(f".{archetype.name}"):
                                return archetype
                            # Match quoted string value: "effect", "compute", etc.
                            if arg_upper == archetype.value.upper():
                                return archetype

            # Check @effect_handler, @reducer_handler patterns
            # Use suffix/prefix matching to reduce false positives from decorators
            # that happen to contain archetype keywords (e.g., @side_effect)
            decorator_name = self._get_name_from_expr(decorator)
            if decorator_name is not None:
                decorator_lower = decorator_name.lower()

                # Check for specific handler decorator patterns:
                # - Suffix: *_effect, *_reducer, *_orchestrator, *_compute
                # - Prefix: effect_*, reducer_*, orchestrator_*, compute_*
                # - Exact: effect, reducer, orchestrator, compute
                #
                # This avoids false positives like:
                # - @side_effect (not an effect handler decorator)
                # - @no_compute (not a compute handler decorator)

                # Effect handler patterns
                if (
                    decorator_lower.endswith("_effect")
                    or decorator_lower.startswith("effect_")
                    or decorator_lower == "effect"
                    or decorator_lower == "effect_handler"
                ):
                    return EnumNodeArchetype.EFFECT

                # Reducer handler patterns
                if (
                    decorator_lower.endswith("_reducer")
                    or decorator_lower.startswith("reducer_")
                    or decorator_lower == "reducer"
                    or decorator_lower == "reducer_handler"
                ):
                    return EnumNodeArchetype.REDUCER

                # Orchestrator handler patterns
                if (
                    decorator_lower.endswith("_orchestrator")
                    or decorator_lower.startswith("orchestrator_")
                    or decorator_lower == "orchestrator"
                    or decorator_lower == "orchestrator_handler"
                ):
                    return EnumNodeArchetype.ORCHESTRATOR

                # Compute handler patterns
                if (
                    decorator_lower.endswith("_compute")
                    or decorator_lower.startswith("compute_")
                    or decorator_lower == "compute"
                    or decorator_lower == "compute_handler"
                ):
                    return EnumNodeArchetype.COMPUTE

        return None

    def _get_name_from_expr(self, expr: ast.expr) -> str | None:
        """Extract a name string from an AST expression.

        Handles:
            - Name nodes: x -> "x"
            - Attribute nodes: a.b.c -> "a.b.c"

        Args:
            expr: The AST expression.

        Returns:
            The name as a string, or None if cannot be extracted.
        """
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            base = self._get_name_from_expr(expr.value)
            if base is not None:
                return f"{base}.{expr.attr}"
            return expr.attr
        return None

    def _validate_handler(
        self,
        handler: ModelDetectedNodeInfo,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Validate a single handler for execution shape violations.

        Args:
            handler: The handler information.

        Returns:
            List of violations found in this handler.
        """
        violations: list[ModelExecutionShapeViolationResult] = []
        rule = self._rules.get(handler.node_archetype)

        if rule is None:
            return violations

        # Get all methods if this is a class
        methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        if isinstance(handler.node, ast.ClassDef):
            for item in handler.node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    methods.append(item)
        else:
            methods = [handler.node]

        # Validate each method
        for method in methods:
            # Check return type violations
            return_violations = self._check_return_type_violations(
                method, handler, rule
            )
            violations.extend(return_violations)

            # Check direct publish violations
            publish_violations = self._check_direct_publish_violations(method, handler)
            violations.extend(publish_violations)

            # Check system time access violations (reducers only)
            if not rule.can_access_system_time:
                time_violations = self._check_system_time_violations(method, handler)
                violations.extend(time_violations)

        return violations

    def _check_return_type_violations(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        handler: ModelDetectedNodeInfo,
        rule: ModelExecutionShapeRule,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Check for forbidden return type violations.

        Analyzes:
            1. Return type annotations
            2. Actual return statements

        Args:
            method: The method AST node.
            handler: The handler information.
            rule: The execution shape rule to validate against.

        Returns:
            List of return type violations.
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        # Check return type annotation
        if method.returns is not None:
            annotation_str = self._get_name_from_expr(method.returns)
            if annotation_str is not None:
                category = self._detect_message_category(annotation_str)
                if category is not None and not self._is_return_type_allowed(
                    category, handler.node_archetype, rule
                ):
                    violation = self._create_return_type_violation(
                        handler, method.lineno, category, annotation_str
                    )
                    if violation is not None:
                        violations.append(violation)

        # Check actual return statements
        for node in ast.walk(method):
            if isinstance(node, ast.Return) and node.value is not None:
                return_violations = self._analyze_return_value(node, handler, rule)
                violations.extend(return_violations)

        return violations

    def _is_return_type_allowed(
        self,
        category: EnumMessageCategory | EnumNodeOutputType,
        node_archetype: EnumNodeArchetype,
        rule: ModelExecutionShapeRule,
    ) -> bool:
        """Check if a return type is allowed for a handler.

        Handles both EnumMessageCategory (EVENT, COMMAND, INTENT) and
        EnumNodeOutputType (PROJECTION) appropriately by converting
        message categories to their corresponding node output types.

        Uses the module-level _MESSAGE_CATEGORY_TO_OUTPUT_TYPE mapping for
        type conversion. See that constant's docstring for forward
        compatibility notes.

        Args:
            category: The detected message category or node output type.
            node_archetype: The node archetype to check against.
            rule: The execution shape rule (uses EnumNodeOutputType).

        Returns:
            True if the return type is allowed, False otherwise.
        """
        # Convert EnumMessageCategory to EnumNodeOutputType for validation
        # since the rules use EnumNodeOutputType consistently
        output_type: EnumNodeOutputType
        if isinstance(category, EnumNodeOutputType):
            output_type = category
        elif isinstance(category, EnumMessageCategory):
            # Use module-level mapping constant for type conversion
            mapped_type = _MESSAGE_CATEGORY_TO_OUTPUT_TYPE.get(category)
            if mapped_type is None:
                # Forward compatibility: Unknown/new message category - disallow
                # by default until explicitly added to the module-level mapping.
                return False
            output_type = mapped_type
        else:
            # Unknown category type - disallow by default
            return False

        return rule.is_return_type_allowed(output_type)

    def _analyze_return_value(
        self,
        return_node: ast.Return,
        handler: ModelDetectedNodeInfo,
        rule: ModelExecutionShapeRule,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Analyze a return statement for forbidden types.

        Args:
            return_node: The return statement AST node.
            handler: The handler information.
            rule: The execution shape rule.

        Returns:
            List of violations from this return statement.
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        if return_node.value is None:
            return violations

        # Check if returning a call like Event(...) or ModelEvent(...)
        if isinstance(return_node.value, ast.Call):
            func_name = self._get_name_from_expr(return_node.value.func)
            if func_name is not None:
                category = self._detect_message_category(func_name)
                if category is not None and not self._is_return_type_allowed(
                    category, handler.node_archetype, rule
                ):
                    violation = self._create_return_type_violation(
                        handler, return_node.lineno, category, func_name
                    )
                    if violation is not None:
                        violations.append(violation)

        # Check if returning a variable with a type-hinted name
        if isinstance(return_node.value, ast.Name):
            var_name = return_node.value.id
            category = self._detect_message_category(var_name)
            if category is not None and not self._is_return_type_allowed(
                category, handler.node_archetype, rule
            ):
                violation = self._create_return_type_violation(
                    handler, return_node.lineno, category, var_name
                )
                if violation is not None:
                    violations.append(violation)

        return violations

    def _detect_message_category(
        self, name: str
    ) -> EnumMessageCategory | EnumNodeOutputType | None:
        """Detect message category or node output type from a type or variable name.

        Uses a multi-phase detection strategy:
        1. First, check for exact suffix matches (most reliable, fewest false positives)
        2. Then, check for prefix patterns (Model* naming convention)
        3. Check for exact uppercase enum-style names
        4. Finally, lenient substring matching (catches non-standard names)

        Note:
            PROJECTION returns EnumNodeOutputType.PROJECTION because projections
            are node output types, not message categories for routing. Projections
            are produced by REDUCER nodes but are not routed via Kafka topics.

        KNOWN LIMITATIONS:
        - Substring matching in phase 4 may produce false positives for names that
          contain message keywords but aren't message types (e.g., "PreventEvent",
          "CommandLineParser", "UserIntent" as a business object).
        - The suffix-based approach is preferred for ONEX-compliant code.

        Args:
            name: The name to analyze (type name, class name, or variable name).

        Returns:
            The detected message category (EnumMessageCategory) for EVENT/COMMAND/INTENT,
            EnumNodeOutputType.PROJECTION for projections, or None if not a message type.
        """
        # Phase 1: Check suffix-based patterns (most reliable, fewest false positives)
        # Suffix matching is more precise than substring matching because message
        # types conventionally END with their category: OrderCreatedEvent, CreateOrderCommand
        #
        # Check longer suffixes first to avoid partial matches:
        # "EventMessage" should match before "Event"
        suffix_patterns: list[tuple[str, EnumMessageCategory | EnumNodeOutputType]] = [
            # Event suffixes - ordered by length (longest first)
            ("EventMessage", EnumMessageCategory.EVENT),
            ("Event", EnumMessageCategory.EVENT),
            # Command suffixes - ordered by length (longest first)
            ("CommandMessage", EnumMessageCategory.COMMAND),
            ("Command", EnumMessageCategory.COMMAND),
            # Intent suffixes - ordered by length (longest first)
            ("IntentMessage", EnumMessageCategory.INTENT),
            ("Intent", EnumMessageCategory.INTENT),
            # Projection suffixes - use EnumNodeOutputType (not a message category)
            ("ProjectionMessage", EnumNodeOutputType.PROJECTION),
            ("Projection", EnumNodeOutputType.PROJECTION),
        ]

        for suffix, category in suffix_patterns:
            if name.endswith(suffix):
                return category

        # Phase 2: Check prefix patterns for Model* naming convention
        # ONEX models use "Model" prefix: ModelEvent, ModelCommand, etc.
        prefix_patterns: list[tuple[str, EnumMessageCategory | EnumNodeOutputType]] = [
            ("ModelEvent", EnumMessageCategory.EVENT),
            ("ModelCommand", EnumMessageCategory.COMMAND),
            ("ModelIntent", EnumMessageCategory.INTENT),
            ("ModelProjection", EnumNodeOutputType.PROJECTION),
        ]

        for prefix, category in prefix_patterns:
            if name.startswith(prefix):
                return category

        # Phase 3: Check for exact uppercase enum-style names
        # These are typically enum values like EVENT, COMMAND, etc.
        uppercase_patterns: dict[str, EnumMessageCategory | EnumNodeOutputType] = {
            "EVENT": EnumMessageCategory.EVENT,
            "COMMAND": EnumMessageCategory.COMMAND,
            "INTENT": EnumMessageCategory.INTENT,
            "PROJECTION": EnumNodeOutputType.PROJECTION,
        }

        if name in uppercase_patterns:
            return uppercase_patterns[name]

        # Phase 4: Lenient substring matching for non-standard naming
        # This is intentionally the last resort and may produce false positives.
        # Examples of potential false positives:
        # - "prevent_event" (utility function, not event type)
        # - "command_line_args" (CLI argument, not command type)
        # - "user_intent" (business concept, not intent message)
        #
        # We use case-insensitive matching here to catch various conventions.
        name_lower = name.lower()

        # Only match if the keyword appears as a word boundary (reduces false positives)
        # Check for common patterns where the keyword is at the end or preceded by underscore
        if name_lower.endswith("event") or "_event" in name_lower:
            return EnumMessageCategory.EVENT
        if name_lower.endswith("command") or "_command" in name_lower:
            return EnumMessageCategory.COMMAND
        if name_lower.endswith("intent") or "_intent" in name_lower:
            return EnumMessageCategory.INTENT
        if name_lower.endswith("projection") or "_projection" in name_lower:
            return EnumNodeOutputType.PROJECTION

        return None

    def _create_return_type_violation(
        self,
        handler: ModelDetectedNodeInfo,
        line_number: int,
        category: EnumMessageCategory | EnumNodeOutputType,
        type_name: str,
    ) -> ModelExecutionShapeViolationResult | None:
        """Create a return type violation result.

        Args:
            handler: The handler information.
            line_number: Line number of the violation.
            category: The forbidden message category or node output type.
            type_name: The actual type name in code.

        Returns:
            The violation result, or None if no matching violation type.
        """
        violation_type: EnumExecutionShapeViolation | None = None
        message = ""

        # Get the execution shape rule for context
        rule = self._rules.get(handler.node_archetype)
        allowed_types_str = (
            ", ".join(t.name for t in rule.allowed_return_types) if rule else "unknown"
        )

        # Helper to check if category matches EVENT (from either enum)
        def is_event(cat: EnumMessageCategory | EnumNodeOutputType) -> bool:
            return cat == EnumMessageCategory.EVENT or cat == EnumNodeOutputType.EVENT

        # Helper to check if category matches INTENT (from either enum)
        def is_intent(cat: EnumMessageCategory | EnumNodeOutputType) -> bool:
            return cat == EnumMessageCategory.INTENT or cat == EnumNodeOutputType.INTENT

        # Helper to check if category matches PROJECTION
        def is_projection(cat: EnumMessageCategory | EnumNodeOutputType) -> bool:
            return cat == EnumNodeOutputType.PROJECTION

        # Get the category name for consistent error messages
        category_name = category.name if hasattr(category, "name") else str(category)

        if handler.node_archetype == EnumNodeArchetype.REDUCER:
            if is_event(category):
                violation_type = EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS
                message = (
                    f"Execution shape violation: Node archetype 'REDUCER' cannot produce "
                    f"output type 'EVENT'. Handler '{handler.name}' returns type '{type_name}'. "
                    f"Expected output types: [{allowed_types_str}]. Found: {category_name}. "
                    f"Reducers must be pure state projectors; event emission is an EFFECT operation."
                )
        elif handler.node_archetype == EnumNodeArchetype.ORCHESTRATOR:
            if is_intent(category):
                violation_type = (
                    EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_INTENTS
                )
                message = (
                    f"Execution shape violation: Node archetype 'ORCHESTRATOR' cannot produce "
                    f"output type 'INTENT'. Handler '{handler.name}' returns type '{type_name}'. "
                    f"Expected output types: [{allowed_types_str}]. Found: {category_name}. "
                    f"Intents represent external user/system requests and should not be "
                    f"generated by orchestration logic."
                )
            elif is_projection(category):
                violation_type = (
                    EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_PROJECTIONS
                )
                message = (
                    f"Execution shape violation: Node archetype 'ORCHESTRATOR' cannot produce "
                    f"output type 'PROJECTION'. Handler '{handler.name}' returns type '{type_name}'. "
                    f"Expected output types: [{allowed_types_str}]. Found: {category_name}. "
                    f"PROJECTION (EnumNodeOutputType.PROJECTION) represents aggregated state "
                    f"and is only valid for REDUCER nodes. Orchestrators coordinate workflows "
                    f"and should emit COMMAND or EVENT types."
                )
        elif handler.node_archetype == EnumNodeArchetype.EFFECT:
            if is_projection(category):
                violation_type = EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
                message = (
                    f"Execution shape violation: Node archetype 'EFFECT' cannot produce "
                    f"output type 'PROJECTION'. Handler '{handler.name}' returns type '{type_name}'. "
                    f"Expected output types: [{allowed_types_str}]. Found: {category_name}. "
                    f"PROJECTION (EnumNodeOutputType.PROJECTION) represents aggregated state "
                    f"and is only valid for REDUCER nodes. Effect handlers interact with "
                    f"external systems and should emit EVENT or COMMAND types."
                )

        if violation_type is None:
            return None

        return ModelExecutionShapeViolationResult(
            violation_type=violation_type,
            node_archetype=handler.node_archetype,
            file_path=handler.file_path,
            line_number=line_number,
            message=message,
            severity=EnumValidationSeverity.ERROR,
        )

    def _check_direct_publish_violations(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        handler: ModelDetectedNodeInfo,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Check for direct publish method calls.

        Forbidden patterns:
            - self.publish(...)
            - event_bus.publish(...)
            - self.send_event(...)
            - self.emit(...)

        Args:
            method: The method AST node.
            handler: The handler information.

        Returns:
            List of direct publish violations.
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                # Check for method calls like x.publish()
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name in _FORBIDDEN_PUBLISH_METHODS:
                        archetype_name = handler.node_archetype.name
                        violations.append(
                            ModelExecutionShapeViolationResult(
                                violation_type=EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH,
                                node_archetype=handler.node_archetype,
                                file_path=handler.file_path,
                                line_number=node.lineno,
                                message=(
                                    f"Execution shape violation: Node archetype '{archetype_name}' "
                                    f"handler '{handler.name}' calls forbidden method '.{method_name}()' "
                                    f"directly at line {node.lineno}. Direct publish methods are "
                                    f"forbidden for all node archetypes. Allowed: Return message objects "
                                    f"and let the dispatcher handle routing. Forbidden methods: "
                                    f"{', '.join(sorted(_FORBIDDEN_PUBLISH_METHODS))}."
                                ),
                                severity=EnumValidationSeverity.ERROR,
                            )
                        )

        return violations

    def _check_system_time_violations(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        handler: ModelDetectedNodeInfo,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Check for system time access in reducers.

        Forbidden patterns:
            - time.time()
            - datetime.now()
            - datetime.utcnow()
            - datetime.datetime.now()
            - datetime.datetime.utcnow()

        Args:
            method: The method AST node.
            handler: The handler information.

        Returns:
            List of system time access violations.
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                func_name = self._get_name_from_expr(node.func)
                if func_name is not None:
                    # Check against known system time patterns
                    if func_name in _SYSTEM_TIME_PATTERNS:
                        violations.append(
                            ModelExecutionShapeViolationResult(
                                violation_type=EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME,
                                node_archetype=handler.node_archetype,
                                file_path=handler.file_path,
                                line_number=node.lineno,
                                message=(
                                    f"Execution shape violation: Node archetype 'REDUCER' "
                                    f"cannot access system time. Handler '{handler.name}' calls "
                                    f"'{func_name}()' at line {node.lineno}. REDUCER nodes must "
                                    f"be deterministic for event replay consistency. Use timestamps "
                                    f"from the event payload instead of system time. "
                                    f"Forbidden patterns: {', '.join(sorted(_SYSTEM_TIME_PATTERNS))}."
                                ),
                                severity=EnumValidationSeverity.ERROR,
                            )
                        )
                    # Check for method calls on datetime/time modules
                    elif isinstance(node.func, ast.Attribute):
                        attr_name = node.func.attr
                        if attr_name in _TIME_FUNCTION_NAMES:
                            base_name = self._get_name_from_expr(node.func.value)
                            if base_name is not None and (
                                "datetime" in base_name or "time" in base_name
                            ):
                                full_name = f"{base_name}.{attr_name}"
                                violations.append(
                                    ModelExecutionShapeViolationResult(
                                        violation_type=EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME,
                                        node_archetype=handler.node_archetype,
                                        file_path=handler.file_path,
                                        line_number=node.lineno,
                                        message=(
                                            f"Execution shape violation: Node archetype 'REDUCER' "
                                            f"cannot access system time. Handler '{handler.name}' calls "
                                            f"'{full_name}()' at line {node.lineno}. REDUCER nodes must "
                                            f"be deterministic for event replay consistency. Use timestamps "
                                            f"from the event payload instead of system time."
                                        ),
                                        severity=EnumValidationSeverity.ERROR,
                                    )
                                )

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
