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

Usage:
    >>> from omnibase_infra.validation.execution_shape_validator import (
    ...     validate_execution_shapes,
    ...     validate_execution_shapes_ci,
    ... )
    >>> violations = validate_execution_shapes(Path("src/handlers"))
    >>> passed, violations = validate_execution_shapes_ci(Path("src/handlers"))
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.validation.model_execution_shape_rule import (
    ModelExecutionShapeRule,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)

logger = logging.getLogger(__name__)

# Handler type detection patterns
_HANDLER_SUFFIX_MAP: dict[str, EnumHandlerType] = {
    "EffectHandler": EnumHandlerType.EFFECT,
    "ReducerHandler": EnumHandlerType.REDUCER,
    "OrchestratorHandler": EnumHandlerType.ORCHESTRATOR,
    "ComputeHandler": EnumHandlerType.COMPUTE,
    # Base class patterns
    "Effect": EnumHandlerType.EFFECT,
    "Reducer": EnumHandlerType.REDUCER,
    "Orchestrator": EnumHandlerType.ORCHESTRATOR,
    "Compute": EnumHandlerType.COMPUTE,
}

# Patterns for detecting message category types in return annotations
_MESSAGE_CATEGORY_PATTERNS: dict[str, EnumMessageCategory] = {
    # Event patterns
    "Event": EnumMessageCategory.EVENT,
    "EVENT": EnumMessageCategory.EVENT,
    "ModelEvent": EnumMessageCategory.EVENT,
    "EventMessage": EnumMessageCategory.EVENT,
    # Command patterns
    "Command": EnumMessageCategory.COMMAND,
    "COMMAND": EnumMessageCategory.COMMAND,
    "ModelCommand": EnumMessageCategory.COMMAND,
    "CommandMessage": EnumMessageCategory.COMMAND,
    # Intent patterns
    "Intent": EnumMessageCategory.INTENT,
    "INTENT": EnumMessageCategory.INTENT,
    "ModelIntent": EnumMessageCategory.INTENT,
    "IntentMessage": EnumMessageCategory.INTENT,
    # Projection patterns
    "Projection": EnumMessageCategory.PROJECTION,
    "PROJECTION": EnumMessageCategory.PROJECTION,
    "ModelProjection": EnumMessageCategory.PROJECTION,
    "ProjectionMessage": EnumMessageCategory.PROJECTION,
}

# Forbidden direct publish method names
_FORBIDDEN_PUBLISH_METHODS: frozenset[str] = frozenset({
    "publish",
    "send_event",
    "emit",
    "emit_event",
    "dispatch",
    "dispatch_event",
})

# System time access patterns (non-deterministic for reducers)
_SYSTEM_TIME_PATTERNS: frozenset[str] = frozenset({
    "time.time",
    "datetime.now",
    "datetime.utcnow",
    "datetime.datetime.now",
    "datetime.datetime.utcnow",
})

# Module-level time function patterns
_TIME_FUNCTION_NAMES: frozenset[str] = frozenset({
    "time",
    "now",
    "utcnow",
})

# Canonical execution shape rules for each handler type
EXECUTION_SHAPE_RULES: dict[EnumHandlerType, ModelExecutionShapeRule] = {
    EnumHandlerType.EFFECT: ModelExecutionShapeRule(
        handler_type=EnumHandlerType.EFFECT,
        allowed_return_types=[EnumMessageCategory.EVENT, EnumMessageCategory.COMMAND],
        forbidden_return_types=[EnumMessageCategory.PROJECTION],
        can_publish_directly=False,
        can_access_system_time=True,
    ),
    EnumHandlerType.COMPUTE: ModelExecutionShapeRule(
        handler_type=EnumHandlerType.COMPUTE,
        allowed_return_types=[
            EnumMessageCategory.EVENT,
            EnumMessageCategory.COMMAND,
            EnumMessageCategory.INTENT,
            EnumMessageCategory.PROJECTION,
        ],
        forbidden_return_types=[],
        can_publish_directly=False,
        can_access_system_time=True,
    ),
    EnumHandlerType.REDUCER: ModelExecutionShapeRule(
        handler_type=EnumHandlerType.REDUCER,
        allowed_return_types=[EnumMessageCategory.PROJECTION],
        forbidden_return_types=[EnumMessageCategory.EVENT],
        can_publish_directly=False,
        can_access_system_time=False,
    ),
    EnumHandlerType.ORCHESTRATOR: ModelExecutionShapeRule(
        handler_type=EnumHandlerType.ORCHESTRATOR,
        allowed_return_types=[EnumMessageCategory.COMMAND, EnumMessageCategory.EVENT],
        forbidden_return_types=[
            EnumMessageCategory.INTENT,
            EnumMessageCategory.PROJECTION,
        ],
        can_publish_directly=False,
        can_access_system_time=True,
    ),
}


class HandlerInfo:
    """Information about a detected handler in source code.

    Attributes:
        name: The class or function name.
        handler_type: The detected handler type (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
        node: The AST node representing the handler.
        line_number: The line number where the handler is defined.
        file_path: The absolute path to the source file.
    """

    def __init__(
        self,
        name: str,
        handler_type: EnumHandlerType,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        line_number: int,
        file_path: str,
    ) -> None:
        """Initialize HandlerInfo.

        Args:
            name: The handler name.
            handler_type: The detected handler type.
            node: The AST node.
            line_number: Line number in source.
            file_path: Path to source file.
        """
        self.name = name
        self.handler_type = handler_type
        self.node = node
        self.line_number = line_number
        self.file_path = file_path


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

    Example:
        >>> validator = ExecutionShapeValidator()
        >>> violations = validator.validate_file(Path("src/handlers/my_handler.py"))
        >>> for v in violations:
        ...     print(v.format_for_ci())
    """

    def __init__(self, strict: bool = False) -> None:
        """Initialize the validator.

        Args:
            strict: If True, treat warnings as errors.
        """
        self.strict = strict
        self._rules = EXECUTION_SHAPE_RULES

    def validate_file(self, file_path: Path) -> list[ModelExecutionShapeViolationResult]:
        """Validate a single Python file for execution shape violations.

        Args:
            file_path: Path to the Python file to validate.

        Returns:
            List of detected violations. Empty list if no violations found.

        Raises:
            SyntaxError: If the file contains invalid Python syntax.
            FileNotFoundError: If the file does not exist.
        """
        source = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(
                "Failed to parse file",
                extra={"file": str(file_path), "error": str(e)},
            )
            raise

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
        """
        violations: list[ModelExecutionShapeViolationResult] = []
        pattern = "**/*.py" if recursive else "*.py"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and not file_path.name.startswith("_"):
                try:
                    file_violations = self.validate_file(file_path)
                    violations.extend(file_violations)
                except SyntaxError:
                    # Already logged in validate_file
                    continue
                except Exception as e:
                    logger.warning(
                        "Failed to validate file",
                        extra={"file": str(file_path), "error": str(e)},
                    )
                    continue

        return violations

    def _find_handlers(self, tree: ast.AST, file_path: str) -> list[HandlerInfo]:
        """Find all handler classes and functions in an AST.

        Detection methods:
            1. Class name suffix: *EffectHandler, *ReducerHandler, etc.
            2. Base class: class MyHandler(EffectHandler)
            3. Decorator: @handler_type(EnumHandlerType.EFFECT)

        Args:
            tree: The parsed AST.
            file_path: Path to the source file.

        Returns:
            List of detected handlers with their type information.
        """
        handlers: list[HandlerInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                handler_type = self._detect_handler_type_from_class(node)
                if handler_type is not None:
                    handlers.append(
                        HandlerInfo(
                            name=node.name,
                            handler_type=handler_type,
                            node=node,
                            line_number=node.lineno,
                            file_path=file_path,
                        )
                    )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                handler_type = self._detect_handler_type_from_decorator(node)
                if handler_type is not None:
                    handlers.append(
                        HandlerInfo(
                            name=node.name,
                            handler_type=handler_type,
                            node=node,
                            line_number=node.lineno,
                            file_path=file_path,
                        )
                    )

        return handlers

    def _detect_handler_type_from_class(
        self,
        node: ast.ClassDef,
    ) -> EnumHandlerType | None:
        """Detect handler type from class definition.

        Checks:
            1. Class name suffix (e.g., OrderEffectHandler)
            2. Base class names (e.g., class OrderHandler(EffectHandler))
            3. Decorator on class

        Args:
            node: The class definition AST node.

        Returns:
            The detected handler type, or None if not a handler.
        """
        # Check class name suffix
        for suffix, handler_type in _HANDLER_SUFFIX_MAP.items():
            if node.name.endswith(suffix):
                return handler_type

        # Check base classes
        for base in node.bases:
            base_name = self._get_name_from_expr(base)
            if base_name is not None:
                for pattern, handler_type in _HANDLER_SUFFIX_MAP.items():
                    if base_name.endswith(pattern):
                        return handler_type

        # Check decorators
        return self._detect_handler_type_from_decorator(node)

    def _detect_handler_type_from_decorator(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> EnumHandlerType | None:
        """Detect handler type from decorators.

        Looks for patterns like:
            @handler_type(EnumHandlerType.EFFECT)
            @effect_handler
            @reducer_handler

        Args:
            node: The AST node with decorators.

        Returns:
            The detected handler type, or None if not found.
        """
        for decorator in node.decorator_list:
            # Check @handler_type(EnumHandlerType.X) pattern
            if isinstance(decorator, ast.Call):
                func = decorator.func
                func_name = self._get_name_from_expr(func)
                if func_name == "handler_type" and decorator.args:
                    arg = decorator.args[0]
                    arg_str = self._get_name_from_expr(arg)
                    if arg_str is not None:
                        # Handle EnumHandlerType.EFFECT or just EFFECT
                        for handler_type in EnumHandlerType:
                            if handler_type.value in arg_str.lower():
                                return handler_type

            # Check @effect_handler, @reducer_handler patterns
            decorator_name = self._get_name_from_expr(decorator)
            if decorator_name is not None:
                decorator_lower = decorator_name.lower()
                if "effect" in decorator_lower:
                    return EnumHandlerType.EFFECT
                if "reducer" in decorator_lower:
                    return EnumHandlerType.REDUCER
                if "orchestrator" in decorator_lower:
                    return EnumHandlerType.ORCHESTRATOR
                if "compute" in decorator_lower:
                    return EnumHandlerType.COMPUTE

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
        handler: HandlerInfo,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Validate a single handler for execution shape violations.

        Args:
            handler: The handler information.

        Returns:
            List of violations found in this handler.
        """
        violations: list[ModelExecutionShapeViolationResult] = []
        rule = self._rules.get(handler.handler_type)

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
        handler: HandlerInfo,
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
                if category is not None and not rule.is_return_type_allowed(category):
                    violation = self._create_return_type_violation(
                        handler, method.lineno, category, annotation_str
                    )
                    if violation is not None:
                        violations.append(violation)

        # Check actual return statements
        for node in ast.walk(method):
            if isinstance(node, ast.Return) and node.value is not None:
                return_violations = self._analyze_return_value(
                    node, handler, rule
                )
                violations.extend(return_violations)

        return violations

    def _analyze_return_value(
        self,
        return_node: ast.Return,
        handler: HandlerInfo,
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
                if category is not None and not rule.is_return_type_allowed(category):
                    violation = self._create_return_type_violation(
                        handler, return_node.lineno, category, func_name
                    )
                    if violation is not None:
                        violations.append(violation)

        # Check if returning a variable with a type-hinted name
        if isinstance(return_node.value, ast.Name):
            var_name = return_node.value.id
            category = self._detect_message_category(var_name)
            if category is not None and not rule.is_return_type_allowed(category):
                violation = self._create_return_type_violation(
                    handler, return_node.lineno, category, var_name
                )
                if violation is not None:
                    violations.append(violation)

        return violations

    def _detect_message_category(self, name: str) -> EnumMessageCategory | None:
        """Detect message category from a type or variable name.

        Args:
            name: The name to analyze.

        Returns:
            The detected message category, or None if not a message type.
        """
        # Check direct pattern matches
        for pattern, category in _MESSAGE_CATEGORY_PATTERNS.items():
            if pattern in name:
                return category

        # Check regex patterns for common naming conventions
        name_lower = name.lower()
        if re.search(r"event", name_lower):
            return EnumMessageCategory.EVENT
        if re.search(r"command", name_lower):
            return EnumMessageCategory.COMMAND
        if re.search(r"intent", name_lower):
            return EnumMessageCategory.INTENT
        if re.search(r"projection", name_lower):
            return EnumMessageCategory.PROJECTION

        return None

    def _create_return_type_violation(
        self,
        handler: HandlerInfo,
        line_number: int,
        category: EnumMessageCategory,
        type_name: str,
    ) -> ModelExecutionShapeViolationResult | None:
        """Create a return type violation result.

        Args:
            handler: The handler information.
            line_number: Line number of the violation.
            category: The forbidden message category.
            type_name: The actual type name in code.

        Returns:
            The violation result, or None if no matching violation type.
        """
        violation_type: EnumExecutionShapeViolation | None = None
        message = ""

        if handler.handler_type == EnumHandlerType.REDUCER:
            if category == EnumMessageCategory.EVENT:
                violation_type = EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS
                message = (
                    f"Reducer '{handler.name}' returns EVENT type '{type_name}'. "
                    "Reducers cannot emit events; event emission is an effect operation."
                )
        elif handler.handler_type == EnumHandlerType.ORCHESTRATOR:
            if category == EnumMessageCategory.INTENT:
                violation_type = EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_INTENTS
                message = (
                    f"Orchestrator '{handler.name}' returns INTENT type '{type_name}'. "
                    "Intents originate from external systems, not orchestration."
                )
            elif category == EnumMessageCategory.PROJECTION:
                violation_type = (
                    EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_PROJECTIONS
                )
                message = (
                    f"Orchestrator '{handler.name}' returns PROJECTION type '{type_name}'. "
                    "Projections are the domain of reducers for state management."
                )
        elif handler.handler_type == EnumHandlerType.EFFECT:
            if category == EnumMessageCategory.PROJECTION:
                violation_type = EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
                message = (
                    f"Effect handler '{handler.name}' returns PROJECTION type '{type_name}'. "
                    "Projections are derived state, managed by reducers."
                )

        if violation_type is None:
            return None

        return ModelExecutionShapeViolationResult(
            violation_type=violation_type,
            handler_type=handler.handler_type,
            file_path=handler.file_path,
            line_number=line_number,
            message=message,
            severity="error",
        )

    def _check_direct_publish_violations(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        handler: HandlerInfo,
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
                        violations.append(
                            ModelExecutionShapeViolationResult(
                                violation_type=EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH,
                                handler_type=handler.handler_type,
                                file_path=handler.file_path,
                                line_number=node.lineno,
                                message=(
                                    f"Handler '{handler.name}' calls '.{method_name}()' directly. "
                                    "All message routing must go through the event bus abstraction."
                                ),
                                severity="error",
                            )
                        )

        return violations

    def _check_system_time_violations(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        handler: HandlerInfo,
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
                                handler_type=handler.handler_type,
                                file_path=handler.file_path,
                                line_number=node.lineno,
                                message=(
                                    f"Reducer '{handler.name}' calls '{func_name}()'. "
                                    "Reducers must be deterministic for replay consistency."
                                ),
                                severity="error",
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
                                        handler_type=handler.handler_type,
                                        file_path=handler.file_path,
                                        line_number=node.lineno,
                                        message=(
                                            f"Reducer '{handler.name}' calls '{full_name}()'. "
                                            "Reducers must be deterministic for replay consistency."
                                        ),
                                        severity="error",
                                    )
                                )

        return violations


def validate_execution_shapes(
    directory: Path,
    strict: bool = False,
) -> list[ModelExecutionShapeViolationResult]:
    """Validate all Python files in directory for execution shape violations.

    This is the main entry point for batch validation of handler files.

    Args:
        directory: Path to the directory containing handler files.
        strict: If True, treat warnings as errors.

    Returns:
        List of all detected violations across all files.

    Example:
        >>> violations = validate_execution_shapes(Path("src/handlers"))
        >>> for v in violations:
        ...     print(f"{v.file_path}:{v.line_number}: {v.message}")
    """
    validator = ExecutionShapeValidator(strict=strict)
    return validator.validate_directory(directory)


def validate_execution_shapes_ci(
    directory: Path,
) -> tuple[bool, list[ModelExecutionShapeViolationResult]]:
    """CI gate that returns (passed, violations).

    This function is designed for CI pipeline integration. It returns a boolean
    indicating whether the validation passed (no blocking violations) along with
    the full list of violations for reporting.

    Args:
        directory: Path to the directory to validate.

    Returns:
        Tuple of (passed, violations) where:
            - passed: True if no error-severity violations found.
            - violations: Complete list of all violations for reporting.

    Example:
        >>> passed, violations = validate_execution_shapes_ci(Path("src/handlers"))
        >>> if not passed:
        ...     for v in violations:
        ...         print(v.format_for_ci())
        ...     sys.exit(1)
    """
    violations = validate_execution_shapes(directory, strict=True)

    # Check if any blocking violations exist
    has_blocking = any(v.is_blocking() for v in violations)

    return (not has_blocking, violations)


def get_execution_shape_rules() -> dict[EnumHandlerType, ModelExecutionShapeRule]:
    """Get the canonical execution shape rules.

    Returns:
        Dictionary mapping handler types to their execution shape rules.
    """
    return EXECUTION_SHAPE_RULES.copy()


__all__ = [
    "ExecutionShapeValidator",
    "HandlerInfo",
    "validate_execution_shapes",
    "validate_execution_shapes_ci",
    "get_execution_shape_rules",
    "EXECUTION_SHAPE_RULES",
]
