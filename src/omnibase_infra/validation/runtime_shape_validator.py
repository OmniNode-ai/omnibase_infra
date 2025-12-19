# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Shape Validator for ONEX Handler Output Validation.

This module provides runtime validation of handler outputs against the
ONEX 4-node architecture execution shape constraints. It ensures that
handlers only produce message types that are allowed for their handler type.

Scope and Design:
    This is a RUNTIME validator, not static analysis. It operates under these
    assumptions:

    1. **Syntactically Valid Code**: Code has already been parsed by Python's
       interpreter. Syntax errors are detected at import/compile time before
       this validator runs.

    2. **Executing Handler Context**: The validator operates on actual return
       values from executing handlers, not on AST representations.

    3. **Known Handler Type**: Handler type is explicitly declared via the
       @enforce_execution_shape decorator or passed to validation methods.

    **Complementary Validators**:
    - execution_shape_validator.py: AST-based static analysis (catches issues
      before runtime, analyzes code structure)
    - runtime_shape_validator.py (this module): Runtime validation (catches
      issues when actual values are produced)

    The runtime validator catches violations that cannot be determined statically,
    such as dynamically constructed return values or conditional returns.

Execution Shape Rules (imported from execution_shape_validator):
    - EFFECT: Can return EVENTs and COMMANDs, but not PROJECTIONs
    - COMPUTE: Can return any message type (most permissive)
    - REDUCER: Can return PROJECTIONs only, not EVENTs (deterministic state management)
    - ORCHESTRATOR: Can return COMMANDs and EVENTs, but not INTENTs or PROJECTIONs

Usage:
    >>> from omnibase_infra.validation.runtime_shape_validator import (
    ...     RuntimeShapeValidator,
    ...     enforce_execution_shape,
    ... )
    >>> from omnibase_infra.enums import EnumHandlerType, EnumMessageCategory
    >>>
    >>> # Direct validation
    >>> validator = RuntimeShapeValidator()
    >>> violation = validator.validate_handler_output(
    ...     handler_type=EnumHandlerType.REDUCER,
    ...     output=some_event,
    ...     output_category=EnumMessageCategory.EVENT,
    ... )
    >>> if violation:
    ...     print(f"Violation: {violation.message}")
    >>>
    >>> # Decorator usage
    >>> @enforce_execution_shape(EnumHandlerType.REDUCER)
    ... def my_reducer_handler(event):
    ...     return ProjectionResult(...)  # OK
    ...     # return EventResult(...)  # Would raise ExecutionShapeViolationError

Exception:
    ExecutionShapeViolationError: Raised when handler output violates
        execution shape constraints. Contains the full violation details
        in the `violation` attribute.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TypeVar

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.models.errors.model_onex_error import ModelOnexError

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

# Import canonical execution shape rules from execution_shape_validator (single source of truth)
from omnibase_infra.validation.execution_shape_validator import (
    EXECUTION_SHAPE_RULES,
)

# Type variable for generic function signature preservation
F = TypeVar("F", bound=Callable[..., object])


# ==============================================================================
# Violation Type Mapping
# ==============================================================================

# Maps (handler_type, forbidden_category) to specific violation type.
#
# This mapping covers explicitly forbidden return types from EXECUTION_SHAPE_RULES:
#   - EFFECT: Cannot return PROJECTION (explicit forbidden)
#   - REDUCER: Cannot return EVENT (explicit forbidden)
#   - ORCHESTRATOR: Cannot return INTENT, PROJECTION (explicit forbidden)
#
# Note: Some categories are implicitly forbidden because they're not in the
# allowed_return_types list (allow-list mode). For example:
#   - REDUCER only allows PROJECTION, so COMMAND and INTENT are implicitly forbidden
#   - EFFECT only allows EVENT and COMMAND, so INTENT is implicitly forbidden
#
# These implicit violations use FORBIDDEN_RETURN_TYPE as the fallback, which is
# semantically correct: the handler is returning a type that's not in its allow-list.
_VIOLATION_TYPE_MAP: dict[
    tuple[EnumHandlerType, EnumMessageCategory], EnumExecutionShapeViolation
] = {
    (
        EnumHandlerType.REDUCER,
        EnumMessageCategory.EVENT,
    ): EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS,
    (
        EnumHandlerType.ORCHESTRATOR,
        EnumMessageCategory.INTENT,
    ): EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_INTENTS,
    (
        EnumHandlerType.ORCHESTRATOR,
        EnumMessageCategory.PROJECTION,
    ): EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_PROJECTIONS,
    (
        EnumHandlerType.EFFECT,
        EnumMessageCategory.PROJECTION,
    ): EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS,
}


# ==============================================================================
# Exception Class
# ==============================================================================


class ExecutionShapeViolationError(ModelOnexError):
    """Raised when handler output violates execution shape constraints.

    This error is raised at runtime when a handler produces output that
    is forbidden for its declared handler type. For example, a REDUCER
    handler returning an EVENT message.

    Attributes:
        violation: The full violation result with type, handler, and context.

    Example:
        >>> try:
        ...     validator.validate_and_raise(
        ...         handler_type=EnumHandlerType.REDUCER,
        ...         output=event_output,
        ...         output_category=EnumMessageCategory.EVENT,
        ...     )
        ... except ExecutionShapeViolationError as e:
        ...     print(f"Violation: {e.violation.violation_type}")
        ...     print(f"Message: {e.violation.message}")
    """

    def __init__(self, violation: ModelExecutionShapeViolationResult) -> None:
        """Initialize ExecutionShapeViolationError.

        Args:
            violation: The execution shape violation result with full context.
        """
        self.violation = violation
        # handler_type may be None if the handler type couldn't be determined
        handler_type_value = (
            violation.handler_type.value if violation.handler_type is not None else None
        )
        super().__init__(
            message=violation.message,
            error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            violation_type=violation.violation_type.value,
            handler_type=handler_type_value,
            severity=violation.severity,
            file_path=violation.file_path,
            line_number=violation.line_number,
        )


# ==============================================================================
# Message Category Detection
# ==============================================================================


def detect_message_category(message: object) -> EnumMessageCategory | None:
    """Detect message category from object type or attributes.

    This function attempts to determine the message category of an object
    using several strategies:

    1. Check for explicit `category` attribute (EnumMessageCategory value)
    2. Check for explicit `message_category` attribute
    3. Check type name patterns (Event*, Command*, Intent*, Projection*)

    Args:
        message: The message object to analyze.

    Returns:
        The detected EnumMessageCategory, or None if category cannot be determined.

    Example:
        >>> class OrderCreatedEvent:
        ...     pass
        >>> detect_message_category(OrderCreatedEvent())
        EnumMessageCategory.EVENT

        >>> class UserProjection:
        ...     category = EnumMessageCategory.PROJECTION
        >>> detect_message_category(UserProjection())
        EnumMessageCategory.PROJECTION
    """
    if message is None:
        return None

    # Strategy 1: Check for explicit category attribute
    if hasattr(message, "category"):
        category = message.category  # type: ignore[union-attr]
        if isinstance(category, EnumMessageCategory):
            return category

    # Strategy 2: Check for message_category attribute
    if hasattr(message, "message_category"):
        category = message.message_category  # type: ignore[union-attr]
        if isinstance(category, EnumMessageCategory):
            return category

    # Strategy 3: Check type name patterns
    type_name = type(message).__name__

    # Check for common naming patterns
    name_patterns: list[tuple[str, EnumMessageCategory]] = [
        ("Event", EnumMessageCategory.EVENT),
        ("Command", EnumMessageCategory.COMMAND),
        ("Intent", EnumMessageCategory.INTENT),
        ("Projection", EnumMessageCategory.PROJECTION),
    ]

    for pattern, category in name_patterns:
        # Check if type name ends with pattern (e.g., OrderCreatedEvent)
        if type_name.endswith(pattern):
            return category
        # Check if type name starts with pattern (e.g., EventOrderCreated)
        if type_name.startswith(pattern):
            return category

    # Could not determine category
    return None


# ==============================================================================
# Runtime Shape Validator
# ==============================================================================


class RuntimeShapeValidator:
    """Runtime validator for ONEX handler execution shape constraints.

    This validator checks handler outputs at runtime against the ONEX 4-node
    architecture execution shape rules. Each handler type has specific
    constraints on what message categories it can produce.

    Attributes:
        rules: Dictionary mapping handler types to their execution shape rules.

    Example:
        >>> validator = RuntimeShapeValidator()
        >>>
        >>> # Check if output is allowed
        >>> if not validator.is_output_allowed(EnumHandlerType.REDUCER, EnumMessageCategory.EVENT):
        ...     print("Reducer cannot return events!")
        >>>
        >>> # Get full violation details
        >>> violation = validator.validate_handler_output(
        ...     handler_type=EnumHandlerType.REDUCER,
        ...     output=event_output,
        ...     output_category=EnumMessageCategory.EVENT,
        ... )
        >>> if violation:
        ...     print(f"Violation: {violation.format_for_ci()}")
        >>>
        >>> # Raise exception on violation
        >>> validator.validate_and_raise(
        ...     handler_type=EnumHandlerType.REDUCER,
        ...     output=event_output,
        ...     output_category=EnumMessageCategory.EVENT,
        ... )  # Raises ExecutionShapeViolationError
    """

    def __init__(
        self, rules: dict[EnumHandlerType, ModelExecutionShapeRule] | None = None
    ) -> None:
        """Initialize RuntimeShapeValidator.

        Args:
            rules: Optional custom rules dictionary. If not provided,
                uses the default EXECUTION_SHAPE_RULES.
        """
        self.rules = rules if rules is not None else EXECUTION_SHAPE_RULES

    def get_rule(self, handler_type: EnumHandlerType) -> ModelExecutionShapeRule:
        """Get execution shape rule for a handler type.

        Args:
            handler_type: The handler type to get the rule for.

        Returns:
            The ModelExecutionShapeRule for the specified handler type.

        Raises:
            KeyError: If no rule is defined for the handler type.
        """
        if handler_type not in self.rules:
            raise KeyError(f"No execution shape rule defined for: {handler_type.value}")
        return self.rules[handler_type]

    def is_output_allowed(
        self,
        handler_type: EnumHandlerType,
        output_category: EnumMessageCategory,
    ) -> bool:
        """Check if an output category is allowed for a handler type.

        This is a quick check that returns True/False without creating
        a full violation result.

        Args:
            handler_type: The handler type to check.
            output_category: The message category of the output.

        Returns:
            True if the output category is allowed, False if forbidden.

        Security Note (Intentional Fail-Open Design):
            This method returns True (allowing the output) when no rule exists
            for the given handler type. This is an INTENTIONAL design decision,
            not a security vulnerability:

            1. **Extensibility**: New handler types should work by default without
               requiring immediate rule definitions. This prevents blocking valid
               code during handler type evolution.

            2. **Validation vs Security Boundary**: This validator enforces
               architectural constraints (ONEX 4-node patterns), NOT security
               policies. It catches developer errors at build/test time, not
               malicious inputs at runtime.

            3. **Defense in Depth**: Security boundaries should be implemented
               at the infrastructure layer (authentication, authorization,
               input validation) - not in architectural pattern validators.

            4. **Fail-Safe for Unknown Types**: Unknown handler types represent
               future extensions or custom implementations. Blocking them would
               break forward compatibility without security benefit.

            If strict validation is required for security-critical contexts,
            use a fail-closed wrapper or policy decorator.
        """
        try:
            rule = self.get_rule(handler_type)
            return rule.is_return_type_allowed(output_category)
        except KeyError:
            # SECURITY DESIGN: Fail-open for unknown handler types.
            # See docstring "Security Note" for rationale.
            # This is intentional - new handler types should be allowed by default.
            return True

    def validate_handler_output(
        self,
        handler_type: EnumHandlerType,
        output: object,
        output_category: EnumMessageCategory,
        file_path: str = "<runtime>",
        line_number: int = 0,
    ) -> ModelExecutionShapeViolationResult | None:
        """Validate handler output against execution shape constraints.

        Args:
            handler_type: The declared handler type.
            output: The actual output object (used for context in violation message).
            output_category: The message category of the output.
            file_path: Optional file path for violation reporting.
            line_number: Optional line number for violation reporting.

        Returns:
            A ModelExecutionShapeViolationResult if a violation is detected,
            or None if the output is valid.
        """
        if self.is_output_allowed(handler_type, output_category):
            return None

        # Determine specific violation type.
        #
        # The _VIOLATION_TYPE_MAP contains explicit mappings for known forbidden
        # combinations (e.g., REDUCER returning EVENT). The fallback to
        # FORBIDDEN_RETURN_TYPE is used for implicitly forbidden categories
        # that aren't in the handler's allow-list but don't have a specific
        # violation type. Examples:
        #   - REDUCER returning COMMAND (not in allowed_return_types=[PROJECTION])
        #   - EFFECT returning INTENT (not in allowed_return_types=[EVENT, COMMAND])
        #
        # The generic FORBIDDEN_RETURN_TYPE is semantically appropriate for these
        # cases because they represent allow-list violations rather than explicit
        # architectural constraint violations.
        violation_key = (handler_type, output_category)
        violation_type = _VIOLATION_TYPE_MAP.get(
            violation_key,
            EnumExecutionShapeViolation.FORBIDDEN_RETURN_TYPE,
        )

        # Build descriptive message with clear actionable guidance.
        # The message distinguishes between explicit forbidden types (architectural
        # constraint) and implicit forbidden types (not in allow-list).
        output_type_name = type(output).__name__
        handler_name = handler_type.value.upper()
        category_name = output_category.value.upper()

        # Provide context-specific guidance based on whether this is an explicit
        # or implicit violation
        if violation_type == EnumExecutionShapeViolation.FORBIDDEN_RETURN_TYPE:
            # Implicit violation: category not in allow-list
            message = (
                f"{handler_name} handler cannot return {category_name} type "
                f"'{output_type_name}'. {category_name} is not in the allowed "
                f"return types for {handler_name} handlers. "
                f"Check EXECUTION_SHAPE_RULES for allowed message categories."
            )
        else:
            # Explicit violation: known architectural constraint
            message = (
                f"{handler_name} handler cannot return {category_name} type "
                f"'{output_type_name}'. This violates ONEX 4-node architecture "
                f"execution shape constraints ({violation_type.value})."
            )

        return ModelExecutionShapeViolationResult(
            violation_type=violation_type,
            handler_type=handler_type,
            file_path=file_path,
            line_number=line_number if line_number > 0 else 1,
            message=message,
            severity="error",
        )

    def validate_and_raise(
        self,
        handler_type: EnumHandlerType,
        output: object,
        output_category: EnumMessageCategory,
        file_path: str = "<runtime>",
        line_number: int = 0,
    ) -> None:
        """Validate handler output and raise exception if invalid.

        This method performs the same validation as validate_handler_output,
        but raises an ExecutionShapeViolationError if a violation is detected.

        Args:
            handler_type: The declared handler type.
            output: The actual output object.
            output_category: The message category of the output.
            file_path: Optional file path for violation reporting.
            line_number: Optional line number for violation reporting.

        Raises:
            ExecutionShapeViolationError: If the output violates execution
                shape constraints for the handler type.
        """
        violation = self.validate_handler_output(
            handler_type=handler_type,
            output=output,
            output_category=output_category,
            file_path=file_path,
            line_number=line_number,
        )
        if violation is not None:
            raise ExecutionShapeViolationError(violation)


# ==============================================================================
# Global Validator Instance
# ==============================================================================

# Default validator instance for convenience
_default_validator = RuntimeShapeValidator()


# ==============================================================================
# Decorator
# ==============================================================================


def enforce_execution_shape(handler_type: EnumHandlerType) -> Callable[[F], F]:
    """Decorator to enforce execution shape constraints at runtime.

    This decorator wraps a handler function and validates its return value
    against the execution shape rules for the specified handler type.
    If the return value violates the constraints, an ExecutionShapeViolationError
    is raised.

    The decorator attempts to detect the message category of the return value
    using the `detect_message_category` function. If the category cannot be
    determined, no validation is performed (fail-open behavior).

    Args:
        handler_type: The handler type that determines allowed output categories.

    Returns:
        A decorator function that wraps the handler with runtime validation.

    Example:
        >>> @enforce_execution_shape(EnumHandlerType.REDUCER)
        ... def my_reducer(event):
        ...     # This is OK - reducer can return projections
        ...     return UserProjection(user_id=event.user_id)
        >>>
        >>> @enforce_execution_shape(EnumHandlerType.REDUCER)
        ... def bad_reducer(event):
        ...     # This will raise ExecutionShapeViolationError
        ...     return UserCreatedEvent(user_id=event.user_id)

    Note:
        The decorator uses inspect to determine the source file and line
        number of the decorated function for better error reporting.

    Security Note (Intentional Fail-Open Design):
        When the message category cannot be determined from the return value,
        the decorator skips validation and allows the return. This is an
        INTENTIONAL design decision for the following reasons:

        1. **Graceful Handling of Unknown Types**: Not all return types will
           have detectable categories (e.g., primitive types, third-party
           objects, custom domain objects). Blocking these would cause false
           positives and break legitimate code.

        2. **Progressive Adoption**: Teams can adopt execution shape validation
           incrementally. Handlers returning non-categorized types continue
           working while teams add category annotations to their message types.

        3. **Validation Tool, Not Security Gate**: This decorator catches
           architectural mistakes (e.g., reducer returning events), not
           security threats. Unknown categories don't represent attack vectors.

        4. **Type Detection Limitations**: Category detection relies on naming
           conventions and explicit attributes. Overly strict enforcement would
           require all types to implement specific interfaces, which is
           impractical for existing codebases.

        For strict enforcement, ensure all message types either:
        - Have a `category` or `message_category` attribute
        - Follow naming conventions (e.g., `*Event`, `*Command`, `*Projection`)
    """

    def decorator(func: F) -> F:
        # Get source info for better error reporting
        try:
            source_file = inspect.getfile(func)
            source_lines = inspect.getsourcelines(func)
            line_number = source_lines[1] if source_lines else 0
        except (TypeError, OSError):
            source_file = "<unknown>"
            line_number = 0

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            result = func(*args, **kwargs)

            # Detect category of the result
            output_category = detect_message_category(result)

            # SECURITY DESIGN: Fail-open for undetectable message categories.
            # See docstring "Security Note" for detailed rationale.
            # This is intentional - unknown types shouldn't block execution.
            if output_category is None:
                return result

            # Validate against execution shape rules
            _default_validator.validate_and_raise(
                handler_type=handler_type,
                output=result,
                output_category=output_category,
                file_path=source_file,
                line_number=line_number,
            )

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    # Constants
    "EXECUTION_SHAPE_RULES",
    # Exception
    "ExecutionShapeViolationError",
    # Functions
    "detect_message_category",
    "enforce_execution_shape",
    # Validator class
    "RuntimeShapeValidator",
]
