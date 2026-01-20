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
    - validator_execution_shape.py: AST-based static analysis (catches issues
      before runtime, analyzes code structure)
    - validator_runtime_shape.py (this module): Runtime validation (catches
      issues when actual values are produced)

    The runtime validator catches violations that cannot be determined statically,
    such as dynamically constructed return values or conditional returns.

Execution Shape Rules (imported from validator_execution_shape):
    - EFFECT: Can return EVENTs and COMMANDs, but not PROJECTIONs
    - COMPUTE: Can return any message type (most permissive)
    - REDUCER: Can return PROJECTIONs only, not EVENTs (deterministic state management)
    - ORCHESTRATOR: Can return COMMANDs and EVENTs, but not INTENTs or PROJECTIONs

Thread Safety:
    This module uses a module-level singleton validator (`_default_validator`) for
    efficiency. The singleton is thread-safe because:

    - The validator is stateless after initialization (only stores immutable rules)
    - No per-validation state is stored in the validator instance
    - Each validation creates fresh violation result objects (no shared mutable state)
    - The singleton is created at module import time (before any threads)
    - The @enforce_execution_shape decorator uses this singleton safely from any thread

    For custom validation rules, create a new RuntimeShapeValidator instance with
    a custom rules dictionary. For most use cases, the singleton provides optimal
    performance without thread safety concerns.

Security Design (Intentional Fail-Open Architecture):
    This validator uses a FAIL-OPEN design by default. This is an INTENTIONAL
    architectural decision, NOT a security vulnerability. Understanding the
    rationale is critical for proper security architecture.

    **What Fail-Open Means Here**:
    - Unknown handler types: Validation returns True (allowed) instead of blocking
    - Undetectable message categories: Decorator skips validation and allows return
    - Missing rules: No exception raised, output is permitted

    **Why Fail-Open is Correct for This Validator**:

    1. **This is an Architectural Validator, NOT a Security Boundary**:
       The RuntimeShapeValidator enforces ONEX 4-node design patterns (Effect,
       Compute, Reducer, Orchestrator) to catch developer mistakes at build/test
       time. It is NOT designed to prevent malicious inputs or unauthorized access.

    2. **Defense-in-Depth Model**:
       Security boundaries should be implemented at the infrastructure layer,
       not in architectural pattern validators:

       - Authentication: Verify identity at API gateway/ingress (OAuth, JWT, mTLS)
       - Authorization: Enforce permissions in service layer (RBAC, ABAC policies)
       - Input Validation: Sanitize untrusted data at entry points (schema validation)
       - Network Security: Restrict access via firewall rules and service mesh

       This validator operates AFTER these security layers, on trusted internal
       handler outputs, making fail-open safe and appropriate.

    3. **Extensibility and Forward Compatibility**:
       New handler types or message categories should work by default without
       requiring immediate rule definitions. Fail-closed would break valid code
       during handler type evolution and prevent progressive adoption.

    4. **Developer Experience**:
       Teams can adopt execution shape validation incrementally. Handlers
       returning non-categorized types continue working while teams add category
       annotations to their message types.

    **When Strict Validation is Needed**:
    If your use case requires fail-closed behavior (e.g., security-critical
    enforcement in production), implement one of these approaches:

    1. **Fail-Closed Wrapper**::

           def strict_is_output_allowed(node_archetype, output_category) -> bool:
               if node_archetype not in EXECUTION_SHAPE_RULES:
                   return False  # Fail-closed for unknown types
               return validator.is_output_allowed(node_archetype, output_category)

    2. **Policy Decorator**::

           @require_known_category  # Custom decorator that rejects None categories
           @enforce_execution_shape(EnumNodeArchetype.REDUCER)
           def my_strict_handler(event):
               return result

    3. **Pre-Validation Check**::

           from omnibase_infra.models.validation import ModelValidateAndRaiseParams

           category = detect_message_category(result)
           if category is None:
               raise ValueError("All outputs must have detectable categories")
           params = ModelValidateAndRaiseParams(
               node_archetype=node_archetype,
               output=result,
               output_category=category,
           )
           validator.validate_and_raise(params)

    **Security Responsibility Boundaries**:
    - This validator: Architectural pattern enforcement (developer guardrails)
    - Infrastructure layer: Authentication, authorization, input validation
    - Application layer: Business logic validation, access control
    - Network layer: TLS, firewall rules, service mesh policies

Usage:
    >>> from omnibase_infra.validation.validator_runtime_shape import (
    ...     RuntimeShapeValidator,
    ...     enforce_execution_shape,
    ... )
    >>> from omnibase_infra.models.validation import ModelOutputValidationParams
    >>> from omnibase_infra.enums import EnumNodeArchetype, EnumMessageCategory
    >>>
    >>> # Direct validation
    >>> validator = RuntimeShapeValidator()
    >>> params = ModelOutputValidationParams(
    ...     node_archetype=EnumNodeArchetype.REDUCER,
    ...     output=some_event,
    ...     output_category=EnumMessageCategory.EVENT,
    ... )
    >>> violation = validator.validate_handler_output(params)
    >>> if violation:
    ...     print(f"Violation: {violation.message}")
    >>>
    >>> # Decorator usage
    >>> @enforce_execution_shape(EnumNodeArchetype.REDUCER)
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
from typing import TypeVar, cast
from uuid import UUID, uuid4

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import (
    EnumExecutionShapeViolation,
    EnumInfraTransportType,
    EnumMessageCategory,
    EnumNodeArchetype,
    EnumNodeOutputType,
    EnumValidationSeverity,
)
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.validation.model_execution_shape_rule import (
    ModelExecutionShapeRule,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)
from omnibase_infra.models.validation.model_output_validation_params import (
    ModelOutputValidationParams,
)
from omnibase_infra.models.validation.model_validate_and_raise_params import (
    ModelValidateAndRaiseParams,
)
from omnibase_infra.types import MessageOutputCategory

# Import canonical execution shape rules from validator_execution_shape (single source of truth)
from omnibase_infra.validation.validator_execution_shape import (
    EXECUTION_SHAPE_RULES,
)

# Type variable for generic function signature preservation
F = TypeVar("F", bound=Callable[..., object])


# ==============================================================================
# Violation Type Mapping
# ==============================================================================

# Maps (node_archetype, forbidden_category) to specific violation type.
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
    tuple[EnumNodeArchetype, MessageOutputCategory],
    EnumExecutionShapeViolation,
] = {
    (
        EnumNodeArchetype.REDUCER,
        EnumMessageCategory.EVENT,
    ): EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS,
    (
        EnumNodeArchetype.ORCHESTRATOR,
        EnumMessageCategory.INTENT,
    ): EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_INTENTS,
    (
        EnumNodeArchetype.ORCHESTRATOR,
        EnumNodeOutputType.PROJECTION,
    ): EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_PROJECTIONS,
    (
        EnumNodeArchetype.EFFECT,
        EnumNodeOutputType.PROJECTION,
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
        >>> from omnibase_infra.models.validation import ModelValidateAndRaiseParams
        >>> from omnibase_infra.enums import EnumNodeArchetype, EnumMessageCategory
        >>> validator = RuntimeShapeValidator()
        >>> try:
        ...     params = ModelValidateAndRaiseParams(
        ...         node_archetype=EnumNodeArchetype.REDUCER,
        ...         output=event_output,
        ...         output_category=EnumMessageCategory.EVENT,
        ...     )
        ...     validator.validate_and_raise(params)
        ... except ExecutionShapeViolationError as e:
        ...     print(f"Violation: {e.violation.violation_type}")
        ...     print(f"Message: {e.violation.message}")

    Correlation ID Support:
        When a correlation_id is provided, it is included in the error context
        for distributed tracing. This enables tracking validation failures back
        to specific requests in multi-service architectures::

            >>> from uuid import uuid4
            >>> correlation_id = uuid4()
            >>> raise ExecutionShapeViolationError(violation, correlation_id=correlation_id)
    """

    def __init__(
        self,
        violation: ModelExecutionShapeViolationResult,
        correlation_id: UUID | None = None,
    ) -> None:
        """Initialize ExecutionShapeViolationError.

        Args:
            violation: The execution shape violation result with full context.
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, enables tracking this validation failure back
                to specific requests across service boundaries.
        """
        self.violation = violation
        # node_archetype may be None if the archetype couldn't be determined
        node_archetype_value = (
            violation.node_archetype.value
            if violation.node_archetype is not None
            else None
        )
        super().__init__(
            message=violation.message,
            error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            correlation_id=correlation_id,
            violation_type=violation.violation_type.value,
            node_archetype=node_archetype_value,
            severity=violation.severity,
            file_path=violation.file_path,
            line_number=violation.line_number,
        )


# ==============================================================================
# Type Conversion Utilities
# ==============================================================================


# Mapping from EnumMessageCategory to EnumNodeOutputType for validation.
# Both enums share EVENT, COMMAND, INTENT with identical string values.
_MESSAGE_CATEGORY_TO_NODE_OUTPUT: dict[EnumMessageCategory, EnumNodeOutputType] = {
    EnumMessageCategory.EVENT: EnumNodeOutputType.EVENT,
    EnumMessageCategory.COMMAND: EnumNodeOutputType.COMMAND,
    EnumMessageCategory.INTENT: EnumNodeOutputType.INTENT,
}


def _to_node_output_type(
    category: MessageOutputCategory,
) -> EnumNodeOutputType:
    """Convert a message category or node output type to EnumNodeOutputType.

    This utility handles the mapping between EnumMessageCategory (routing)
    and EnumNodeOutputType (validation). Since EVENT, COMMAND, and INTENT
    exist in both enums with the same values, they can be safely converted.

    Args:
        category: Either an EnumMessageCategory or EnumNodeOutputType.

    Returns:
        The corresponding EnumNodeOutputType.

    Raises:
        ValueError: If an EnumMessageCategory cannot be mapped to EnumNodeOutputType
            (should not happen with valid enum values).
    """
    if isinstance(category, EnumNodeOutputType):
        return category

    if isinstance(category, EnumMessageCategory):
        if category in _MESSAGE_CATEGORY_TO_NODE_OUTPUT:
            return _MESSAGE_CATEGORY_TO_NODE_OUTPUT[category]
        # This should never happen with valid EnumMessageCategory values
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="to_node_output_type",
            correlation_id=uuid4(),
        )
        raise ProtocolConfigurationError(
            f"Cannot convert {category} to EnumNodeOutputType",
            context=context,
            category=str(category),
        )

    # Type safety fallback (should not be reached with proper typing)
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="to_node_output_type",
        correlation_id=uuid4(),
    )
    raise ProtocolConfigurationError(
        f"Expected EnumMessageCategory or EnumNodeOutputType, got {type(category)}",
        context=context,
        actual_type=type(category).__name__,
    )


# ==============================================================================
# Message Category Detection
# ==============================================================================


def detect_message_category(
    message: object,
) -> MessageOutputCategory | None:
    """Detect message category or node output type from object type or attributes.

    This function attempts to determine the message category or node output type
    of an object using several strategies:

    1. Check for explicit `category` attribute (EnumMessageCategory or EnumNodeOutputType)
    2. Check for explicit `message_category` or `output_type` attribute
    3. Check type name patterns (Event*, Command*, Intent*, Projection*)

    Note: PROJECTION is detected as EnumNodeOutputType.PROJECTION since projections
    are node output types (used by REDUCERs), not message routing categories.

    Args:
        message: The message object to analyze.

    Returns:
        The detected EnumMessageCategory or EnumNodeOutputType,
        or None if category cannot be determined.

    Example:
        >>> class OrderCreatedEvent:
        ...     pass
        >>> detect_message_category(OrderCreatedEvent())
        EnumMessageCategory.EVENT

        >>> class UserProjection:
        ...     output_type = EnumNodeOutputType.PROJECTION
        >>> detect_message_category(UserProjection())
        EnumNodeOutputType.PROJECTION
    """
    if message is None:
        return None

    # Strategy 1: Check for explicit category attribute
    if hasattr(message, "category"):
        category = message.category
        if isinstance(category, EnumMessageCategory | EnumNodeOutputType):
            return category

    # Strategy 2a: Check for message_category attribute
    if hasattr(message, "message_category"):
        category = message.message_category
        if isinstance(category, EnumMessageCategory):
            return category

    # Strategy 2b: Check for output_type attribute (for node outputs like projections)
    if hasattr(message, "output_type"):
        output_type = message.output_type
        if isinstance(output_type, EnumNodeOutputType):
            return output_type

    # Strategy 3: Check type name patterns
    type_name = type(message).__name__

    # Check for common naming patterns
    # Note: PROJECTION returns EnumNodeOutputType since it's a node output type
    name_patterns: list[tuple[str, MessageOutputCategory]] = [
        ("Event", EnumMessageCategory.EVENT),
        ("Command", EnumMessageCategory.COMMAND),
        ("Intent", EnumMessageCategory.INTENT),
        ("Projection", EnumNodeOutputType.PROJECTION),
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
    architecture execution shape rules. Each node archetype has specific
    constraints on what message categories it can produce.

    Attributes:
        rules: Dictionary mapping node archetypes to their execution shape rules.

    Thread Safety:
        RuntimeShapeValidator instances are stateless after initialization.
        The rules dictionary is immutable and no per-validation state is stored.
        Multiple threads can safely call validate_handler_output() or
        validate_and_raise() on the same validator instance concurrently.

    Performance:
        For repeated validation (e.g., decorated handlers in hot paths), use the
        module-level singleton via `enforce_execution_shape()` decorator or the
        `_default_validator` instance for optimal performance.

        Creating new validator instances is cheap but unnecessary in most cases.
        Create a new instance only if you need custom rules or isolation.

    Example:
        >>> from omnibase_infra.models.validation import (
        ...     ModelOutputValidationParams,
        ...     ModelValidateAndRaiseParams,
        ... )
        >>> validator = RuntimeShapeValidator()
        >>>
        >>> # Check if output is allowed
        >>> if not validator.is_output_allowed(EnumNodeArchetype.REDUCER, EnumMessageCategory.EVENT):
        ...     print("Reducer cannot return events!")
        >>>
        >>> # Get full violation details
        >>> params = ModelOutputValidationParams(
        ...     node_archetype=EnumNodeArchetype.REDUCER,
        ...     output=event_output,
        ...     output_category=EnumMessageCategory.EVENT,
        ... )
        >>> violation = validator.validate_handler_output(params)
        >>> if violation:
        ...     print(f"Violation: {violation.format_for_ci()}")
        >>>
        >>> # Raise exception on violation
        >>> raise_params = ModelValidateAndRaiseParams(
        ...     node_archetype=EnumNodeArchetype.REDUCER,
        ...     output=event_output,
        ...     output_category=EnumMessageCategory.EVENT,
        ... )
        >>> validator.validate_and_raise(raise_params)  # Raises ExecutionShapeViolationError
    """

    def __init__(
        self, rules: dict[EnumNodeArchetype, ModelExecutionShapeRule] | None = None
    ) -> None:
        """Initialize RuntimeShapeValidator.

        Args:
            rules: Optional custom rules dictionary. If not provided,
                uses the default EXECUTION_SHAPE_RULES.
        """
        self.rules = rules if rules is not None else EXECUTION_SHAPE_RULES

    def get_rule(self, node_archetype: EnumNodeArchetype) -> ModelExecutionShapeRule:
        """Get execution shape rule for a node archetype.

        Args:
            node_archetype: The node archetype to get the rule for.

        Returns:
            The ModelExecutionShapeRule for the specified node archetype.

        Raises:
            KeyError: If no rule is defined for the node archetype.
        """
        if node_archetype not in self.rules:
            raise KeyError(
                f"No execution shape rule defined for: {node_archetype.value}"
            )
        return self.rules[node_archetype]

    def _get_allowed_types_for_handler(self, node_archetype: EnumNodeArchetype) -> str:
        """Get a human-readable string of allowed output types for a handler.

        Used to generate helpful error messages that suggest valid alternatives
        when a handler attempts to return a forbidden output type.

        Args:
            node_archetype: The node archetype to get allowed types for.

        Returns:
            A formatted string listing allowed output types (e.g., "EVENTs or COMMANDs").
        """
        try:
            rule = self.get_rule(node_archetype)
            allowed = [t.value.upper() + "s" for t in rule.allowed_return_types]
            if len(allowed) == 0:
                return "no specific types (check EXECUTION_SHAPE_RULES)"
            if len(allowed) == 1:
                return allowed[0]
            if len(allowed) == 2:
                return f"{allowed[0]} or {allowed[1]}"
            return ", ".join(allowed[:-1]) + f", or {allowed[-1]}"
        except KeyError:
            return "appropriate types (check EXECUTION_SHAPE_RULES)"

    def is_output_allowed(
        self,
        node_archetype: EnumNodeArchetype,
        output_category: MessageOutputCategory,
    ) -> bool:
        """Check if an output category is allowed for a node archetype.

        This is a quick check that returns True/False without creating
        a full violation result.

        Args:
            node_archetype: The node archetype to check.
            output_category: The message category or node output type of the output.

        Returns:
            True if the output category is allowed, False if forbidden.

        Security Note (Intentional Fail-Open Design):
            This method returns True (allowing the output) when no rule exists
            for the given node archetype. This is an INTENTIONAL design decision,
            not a security vulnerability:

            1. **Extensibility**: New node archetypes should work by default without
               requiring immediate rule definitions. This prevents blocking valid
               code during archetype evolution.

            2. **Validation vs Security Boundary**: This validator enforces
               architectural constraints (ONEX 4-node patterns), NOT security
               policies. It catches developer errors at build/test time, not
               malicious inputs at runtime.

            3. **Defense in Depth**: Security boundaries should be implemented
               at the infrastructure layer (authentication, authorization,
               input validation) - not in architectural pattern validators.

            4. **Fail-Safe for Unknown Types**: Unknown node archetypes represent
               future extensions or custom implementations. Blocking them would
               break forward compatibility without security benefit.

            If strict validation is required for security-critical contexts,
            use a fail-closed wrapper or policy decorator.
        """
        try:
            rule = self.get_rule(node_archetype)
            # Convert to EnumNodeOutputType for validation
            node_output_type = _to_node_output_type(output_category)
            return rule.is_return_type_allowed(node_output_type)
        except KeyError:
            # SECURITY DESIGN: Fail-open for unknown node archetypes.
            # See docstring "Security Note" for rationale.
            # This is intentional - new archetypes should be allowed by default.
            return True

    def validate_handler_output(
        self,
        params: ModelOutputValidationParams,
    ) -> ModelExecutionShapeViolationResult | None:
        """Validate handler output against execution shape constraints.

        Args:
            params: Parameters encapsulating node_archetype, output, output_category,
                and optional file_path and line_number for violation reporting.

        Returns:
            A ModelExecutionShapeViolationResult if a violation is detected,
            or None if the output is valid.
        """
        if self.is_output_allowed(params.node_archetype, params.output_category):
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
        violation_key = (params.node_archetype, params.output_category)
        violation_type = _VIOLATION_TYPE_MAP.get(
            violation_key,
            EnumExecutionShapeViolation.FORBIDDEN_RETURN_TYPE,
        )

        # Build descriptive message with clear actionable guidance.
        # The message distinguishes between explicit forbidden types (architectural
        # constraint) and implicit forbidden types (not in allow-list).
        output_type_name = type(params.output).__name__
        archetype_name = params.node_archetype.value.upper()
        category_name = params.output_category.value.upper()

        # Check if this is a PROJECTION violation - these need educational context
        # about the distinction between node output types and message categories
        is_projection_violation = (
            params.output_category == EnumNodeOutputType.PROJECTION
        )

        # Provide context-specific guidance based on whether this is an explicit
        # or implicit violation
        if violation_type == EnumExecutionShapeViolation.FORBIDDEN_RETURN_TYPE:
            # Implicit violation: category not in allow-list
            if is_projection_violation:
                # Special case: PROJECTION needs educational message
                message = (
                    f"{archetype_name} handler cannot return PROJECTION type "
                    f"'{output_type_name}'. PROJECTION is a node output type "
                    f"(EnumNodeOutputType), not a message routing category "
                    f"(EnumMessageCategory). Projections represent aggregated state "
                    f"and are only valid for REDUCER node output types."
                )
            else:
                message = (
                    f"{archetype_name} handler cannot return {category_name} type "
                    f"'{output_type_name}'. {category_name} is not in the allowed "
                    f"return types for {archetype_name} handlers. "
                    f"Check EXECUTION_SHAPE_RULES for allowed message categories."
                )
        # Explicit violation: known architectural constraint
        elif is_projection_violation:
            # PROJECTION violations get enhanced educational message
            allowed_types = self._get_allowed_types_for_handler(params.node_archetype)
            message = (
                f"{archetype_name} handler cannot return PROJECTION type "
                f"'{output_type_name}'. PROJECTION is a node output type "
                f"(EnumNodeOutputType), not a message routing category "
                f"(EnumMessageCategory). Projections represent aggregated state "
                f"and are only valid for REDUCER node output types. "
                f"{archetype_name} handlers should return {allowed_types} instead."
            )
        else:
            message = (
                f"{archetype_name} handler cannot return {category_name} type "
                f"'{output_type_name}'. This violates ONEX 4-node architecture "
                f"execution shape constraints ({violation_type.value})."
            )

        return ModelExecutionShapeViolationResult(
            violation_type=violation_type,
            node_archetype=params.node_archetype,
            file_path=params.file_path,
            line_number=params.line_number if params.line_number > 0 else 1,
            message=message,
            severity=EnumValidationSeverity.ERROR,
        )

    def validate_and_raise(
        self,
        params: ModelValidateAndRaiseParams,
    ) -> None:
        """Validate handler output and raise exception if invalid.

        This method performs the same validation as validate_handler_output,
        but raises an ExecutionShapeViolationError if a violation is detected.

        Args:
            params: Parameters encapsulating node_archetype, output, output_category,
                optional file_path and line_number for violation reporting,
                and optional correlation_id for distributed tracing.

        Raises:
            ExecutionShapeViolationError: If the output violates execution
                shape constraints for the node archetype.
        """
        validation_params = ModelOutputValidationParams(
            node_archetype=params.node_archetype,
            output=params.output,
            output_category=params.output_category,
            file_path=params.file_path,
            line_number=params.line_number,
        )
        violation = self.validate_handler_output(validation_params)
        if violation is not None:
            raise ExecutionShapeViolationError(
                violation, correlation_id=params.correlation_id
            )


# ==============================================================================
# Module-Level Singleton Validator
# ==============================================================================
#
# Performance Optimization: The RuntimeShapeValidator is stateless after
# initialization (only stores a reference to EXECUTION_SHAPE_RULES which is
# a module-level constant). Creating new instances on every validation call
# is wasteful in hot paths (e.g., the @enforce_execution_shape decorator).
# Instead, we use a module-level singleton.
#
# Why a singleton is safe here:
# - The validator's rules dictionary is immutable after initialization
# - No per-validation state is stored in the validator instance
# - All validation methods are pure functions that produce new results
# - Each validation creates fresh ModelExecutionShapeViolationResult objects
#
# Thread Safety:
# - The singleton is created at module import time (before any threads)
# - All read operations on the rules dictionary are thread-safe
# - Each validation creates fresh local state (violation results)
# - No locks are needed because there's no shared mutable state
# - The @enforce_execution_shape decorator uses this singleton safely from any thread
# - Concurrent calls from multiple threads will each create independent violation
#   results without interference
#
# When NOT to use the singleton:
# - If you need custom execution shape rules (create your own instance)
# - If you need to mock the validator in tests (inject or patch)
# - If you're validating in a context that requires isolation
#
# For repeated validation (e.g., decorated handlers in hot paths), the singleton
# pattern provides optimal performance without sacrificing thread safety.

_default_validator = RuntimeShapeValidator()


# ==============================================================================
# Decorator
# ==============================================================================


def enforce_execution_shape(node_archetype: EnumNodeArchetype) -> Callable[[F], F]:
    """Decorator to enforce execution shape constraints at runtime.

    This decorator wraps a handler function and validates its return value
    against the execution shape rules for the specified node archetype.
    If the return value violates the constraints, an ExecutionShapeViolationError
    is raised.

    The decorator attempts to detect the message category of the return value
    using the `detect_message_category` function. If the category cannot be
    determined, no validation is performed (fail-open behavior).

    Args:
        node_archetype: The node archetype that determines allowed output categories.

    Returns:
        A decorator function that wraps the handler with runtime validation.

    Example:
        >>> @enforce_execution_shape(EnumNodeArchetype.REDUCER)
        ... def my_reducer(event):
        ...     # This is OK - reducer can return projections
        ...     return UserProjection(user_id=event.user_id)
        >>>
        >>> @enforce_execution_shape(EnumNodeArchetype.REDUCER)
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
            validate_params = ModelValidateAndRaiseParams(
                node_archetype=node_archetype,
                output=result,
                output_category=output_category,
                file_path=source_file,
                line_number=line_number,
            )
            _default_validator.validate_and_raise(validate_params)

            return result

        # Cast wrapper to F - functools.wraps preserves the signature at runtime,
        # and mypy cannot prove the equivalence, so we use an explicit cast.
        return cast("F", wrapper)

    return decorator


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    # Constants
    "EXECUTION_SHAPE_RULES",
    # Exception
    "ExecutionShapeViolationError",
    # Validator class
    "RuntimeShapeValidator",
    # Functions
    "detect_message_category",
    "enforce_execution_shape",
]
