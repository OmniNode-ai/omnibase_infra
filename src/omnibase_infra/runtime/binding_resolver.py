# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Binding expression parser and resolver for operation bindings.

This module provides:

- **BindingExpressionParser**: Parses ${source.path} expressions with guardrails
- **BindingExpressionParseError**: Typed error with error codes for parse failures
- **OperationBindingResolver**: Resolves bindings from envelope/payload/context

Expression Syntax
-----------------
Expressions follow the format: ``${source.path.to.field}``

Where:
- **source** is one of: ``payload``, ``envelope``, ``context``
- **path** is a dot-separated sequence of field names (no array indexing)

Examples::

    ${payload.user.id}        # Resolve from payload.user.id
    ${envelope.correlation_id}  # Resolve from envelope.correlation_id
    ${context.now_iso}        # Resolve from context.now_iso

Guardrails
----------
The parser enforces the following guardrails:

- **Max expression length**: 256 characters
- **Max path depth**: 20 segments
- **No array indexing**: Expressions like ``${payload.items[0]}`` are rejected
- **Valid sources only**: Must be ``payload``, ``envelope``, or ``context``
- **Context path allowlist**: Context paths must be in ``VALID_CONTEXT_PATHS``

Resolution Behavior
-------------------
The resolver follows these rules:

1. Global bindings are applied first
2. Operation-specific bindings override globals for the same parameter
3. Required bindings fail fast if resolved value is None
4. Optional bindings use defaults when resolved value is None

Thread Safety
-------------
Both ``BindingExpressionParser`` and ``OperationBindingResolver`` are stateless
and thread-safe. They can be shared across concurrent requests.

.. versionadded:: 0.2.6
    Created as part of OMN-1518 - Declarative operation bindings.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel

from omnibase_core.types import JsonType
from omnibase_infra.errors import BindingResolutionError
from omnibase_infra.models.bindings import (
    DEFAULT_JSON_RECURSION_DEPTH,
    EXPRESSION_PATTERN,
    MAX_EXPRESSION_LENGTH,
    MAX_PATH_SEGMENTS,
    VALID_CONTEXT_PATHS,
    VALID_SOURCES,
    ModelBindingResolutionResult,
    ModelOperationBindingsSubcontract,
    ModelParsedBinding,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Error Codes for Expression Parsing
# =============================================================================


class EnumBindingParseErrorCode(str, Enum):
    """Error codes for binding expression parse failures.

    These error codes are used by BindingExpressionParseError to provide
    typed error classification. The codes align with the BINDING_LOADER_0xx
    error codes used by the operation bindings loader.

    .. versionadded:: 0.2.6
    """

    EXPRESSION_MALFORMED = "BINDING_LOADER_010"
    """Expression syntax is invalid (missing ${}, wrong delimiters, array access)."""

    INVALID_SOURCE = "BINDING_LOADER_011"
    """Source is not one of: payload, envelope, context."""

    PATH_TOO_DEEP = "BINDING_LOADER_012"
    """Path exceeds MAX_PATH_SEGMENTS (20 segments)."""

    EXPRESSION_TOO_LONG = "BINDING_LOADER_013"
    """Expression exceeds MAX_EXPRESSION_LENGTH (256 characters)."""

    EMPTY_PATH_SEGMENT = "BINDING_LOADER_014"
    """Path contains empty segment (e.g., ${payload..field})."""

    INVALID_CONTEXT_PATH = "BINDING_LOADER_016"
    """Context path is not in VALID_CONTEXT_PATHS allowlist."""


class BindingExpressionParseError(ValueError):
    """Error raised when binding expression parsing fails.

    This exception provides typed error codes for different parse failure
    scenarios, enabling callers to handle errors based on error code rather
    than parsing error message strings.

    Attributes:
        error_code: The specific error code identifying the failure type.
        expression: The expression that failed to parse.
        message: Human-readable error message.

    Example:
        >>> try:
        ...     parser.parse("${invalid.path}")
        ... except BindingExpressionParseError as e:
        ...     if e.error_code == EnumBindingParseErrorCode.INVALID_SOURCE:
        ...         # Handle invalid source
        ...         pass

    .. versionadded:: 0.2.6
    """

    def __init__(
        self,
        message: str,
        error_code: EnumBindingParseErrorCode,
        expression: str,
    ) -> None:
        """Initialize the parse error.

        Args:
            message: Human-readable error description.
            error_code: Typed error code identifying the failure type.
            expression: The expression that failed to parse.
        """
        super().__init__(message)
        self.error_code = error_code
        self.expression = expression
        self.message = message

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


# =============================================================================
# JSON Recursion Depth Limit
# =============================================================================

# NOTE: The default value (DEFAULT_JSON_RECURSION_DEPTH = 100) is now imported
# from omnibase_infra.models.bindings.constants. This enables per-contract
# configuration via ModelOperationBindingsSubcontract.max_json_recursion_depth.
#
# The configurable range is [10, 1000] with sensible defaults:
# - MIN_JSON_RECURSION_DEPTH = 10 (lower values too restrictive)
# - MAX_JSON_RECURSION_DEPTH = 1000 (higher values risk stack overflow)
# - DEFAULT_JSON_RECURSION_DEPTH = 100 (handles normal JSON structures)
#
# .. versionchanged:: 0.2.7
#     Made configurable via contract.yaml. See OMN-1518.


# =============================================================================
# JSON Compatibility Validation
# =============================================================================


def _is_json_compatible(
    value: object,
    max_depth: int = DEFAULT_JSON_RECURSION_DEPTH,
) -> bool:
    """Check if a value is JSON-compatible.

    JSON-compatible values are those that can be serialized to JSON:
    - None
    - Primitives: str, int, float, bool
    - UUID (serializes to string)
    - list (with recursively JSON-compatible elements)
    - dict with str keys (with recursively JSON-compatible values)

    This function performs recursive validation for nested structures.
    It guards against infinite recursion by limiting depth to
    ``max_depth`` levels (default: DEFAULT_JSON_RECURSION_DEPTH = 100).

    Args:
        value: The value to check for JSON compatibility.
        max_depth: Maximum recursion depth for validation. Defaults to
            DEFAULT_JSON_RECURSION_DEPTH (100). Can be configured per-contract
            via ModelOperationBindingsSubcontract.max_json_recursion_depth.

    Returns:
        True if the value is JSON-compatible, False otherwise.

    Examples:
        >>> _is_json_compatible(None)
        True
        >>> _is_json_compatible("hello")
        True
        >>> _is_json_compatible({"key": [1, 2, 3]})
        True
        >>> from uuid import UUID
        >>> _is_json_compatible(UUID("12345678-1234-5678-1234-567812345678"))
        True
        >>> _is_json_compatible(object())
        False
        >>> _is_json_compatible({"key": lambda x: x})
        False

        # With custom depth limit
        >>> _is_json_compatible({"a": {"b": {"c": 1}}}, max_depth=50)
        True

    .. versionadded:: 0.2.6
    .. versionchanged:: 0.2.7
        Added max_depth parameter for per-contract configuration.
    """
    return _is_json_compatible_recursive(value, depth=0, max_depth=max_depth)


def _is_json_compatible_recursive(
    value: object,
    depth: int,
    max_depth: int = DEFAULT_JSON_RECURSION_DEPTH,
) -> bool:
    """Recursive implementation of JSON compatibility check.

    Args:
        value: The value to check.
        depth: Current recursion depth for overflow protection.
        max_depth: Maximum allowed recursion depth.

    Returns:
        True if the value is JSON-compatible, False otherwise.
    """
    # Depth guard to prevent stack overflow on pathological inputs
    if depth > max_depth:
        return False

    # None is JSON-compatible (maps to JSON null)
    if value is None:
        return True

    # JSON primitives
    if isinstance(value, (str, int, float, bool)):
        # Note: bool must be checked with isinstance since bool is a subclass of int
        return True

    # UUID is JSON-compatible (serializes to string)
    if isinstance(value, UUID):
        return True

    # List: all elements must be JSON-compatible
    if isinstance(value, list):
        return all(
            _is_json_compatible_recursive(item, depth + 1, max_depth) for item in value
        )

    # Dict: keys must be str, values must be JSON-compatible
    if isinstance(value, dict):
        return all(
            isinstance(k, str)
            and _is_json_compatible_recursive(v, depth + 1, max_depth)
            for k, v in value.items()
        )

    # Anything else (objects, callables, etc.) is not JSON-compatible
    return False


# =============================================================================
# BindingExpressionParser
# =============================================================================


class BindingExpressionParser:
    """Parse ${source.path} expressions with guardrails.

    This parser validates and decomposes binding expressions into their
    constituent parts: source and path segments. It enforces all guardrails
    at parse time to fail fast on invalid expressions.

    Guardrails enforced (default values, can be overridden per-contract):
        - Max expression length: 256 characters (configurable: 32-1024)
        - Max path depth: 20 segments (configurable: 3-50)
        - No array indexing (``[0]``, ``[*]``)
        - Source must be: ``payload`` | ``envelope`` | ``context``
        - Context paths must be in ``VALID_CONTEXT_PATHS`` or additional_context_paths

    This class is stateless and thread-safe.

    Example:
        >>> parser = BindingExpressionParser()
        >>> source, path = parser.parse("${payload.user.id}")
        >>> source
        'payload'
        >>> path
        ('user', 'id')

        # With custom limits
        >>> source, path = parser.parse(
        ...     "${payload.very.deep.nested.path}",
        ...     max_expression_length=512,
        ...     max_path_segments=30,
        ... )

    .. versionadded:: 0.2.6
    .. versionchanged:: 0.2.7
        Added max_expression_length, max_path_segments, and additional_context_paths
        parameters for per-contract guardrail overrides.
    """

    def parse(
        self,
        expression: str,
        max_expression_length: int | None = None,
        max_path_segments: int | None = None,
        additional_context_paths: frozenset[str] | None = None,
    ) -> tuple[Literal["payload", "envelope", "context"], tuple[str, ...]]:
        """Parse expression into (source, path_segments).

        Args:
            expression: Expression in ``${source.path.to.field}`` format.
            max_expression_length: Override default expression length limit.
                If None, uses MAX_EXPRESSION_LENGTH (256).
            max_path_segments: Override default path segment limit.
                If None, uses MAX_PATH_SEGMENTS (20).
            additional_context_paths: Additional valid context paths beyond the
                base VALID_CONTEXT_PATHS set. If provided, these paths will be
                accepted for context source expressions.

        Returns:
            Tuple of (source, path_segments) where source is one of
            ``"payload"``, ``"envelope"``, ``"context"`` and path_segments
            is a tuple of field names.

        Raises:
            BindingExpressionParseError: If expression is malformed or violates
                any guardrail. The exception includes a typed ``error_code`` attribute
                for programmatic error handling:

                - ``EXPRESSION_TOO_LONG``: Expression exceeds max length
                - ``EXPRESSION_MALFORMED``: Array access or invalid syntax
                - ``INVALID_SOURCE``: Source is not valid
                - ``EMPTY_PATH_SEGMENT``: Path contains empty segments
                - ``PATH_TOO_DEEP``: Path exceeds max segments
                - ``INVALID_CONTEXT_PATH``: Context path is not in allowlist

        Example:
            >>> parser = BindingExpressionParser()
            >>> parser.parse("${payload.user.email}")
            ('payload', ('user', 'email'))

            >>> parser.parse("${context.now_iso}")
            ('context', ('now_iso',))

            >>> parser.parse("${payload.items[0]}")  # Raises BindingExpressionParseError
            Traceback (most recent call last):
                ...
            BindingExpressionParseError: Array access not allowed in expressions: ...

        .. versionchanged:: 0.2.7
            Added max_expression_length, max_path_segments, and additional_context_paths
            parameters for per-contract guardrail overrides.
        """
        # Use configured limits or defaults
        effective_max_length = (
            max_expression_length
            if max_expression_length is not None
            else MAX_EXPRESSION_LENGTH
        )
        effective_max_segments = (
            max_path_segments if max_path_segments is not None else MAX_PATH_SEGMENTS
        )

        # Guardrail: max length
        if len(expression) > effective_max_length:
            raise BindingExpressionParseError(
                f"Expression exceeds max length ({len(expression)} > {effective_max_length})",
                error_code=EnumBindingParseErrorCode.EXPRESSION_TOO_LONG,
                expression=expression,
            )

        # Guardrail: no array access
        if "[" in expression or "]" in expression:
            raise BindingExpressionParseError(
                f"Array access not allowed in expressions: {expression}",
                error_code=EnumBindingParseErrorCode.EXPRESSION_MALFORMED,
                expression=expression,
            )

        # Parse expression
        match = EXPRESSION_PATTERN.match(expression)
        if not match:
            raise BindingExpressionParseError(
                f"Invalid expression syntax: {expression}. "
                f"Expected format: ${{source.path.to.field}}",
                error_code=EnumBindingParseErrorCode.EXPRESSION_MALFORMED,
                expression=expression,
            )

        source = match.group(1)
        path_str = match.group(2)

        # Validate source
        if source not in VALID_SOURCES:
            raise BindingExpressionParseError(
                f"Invalid source '{source}'. Must be one of: {sorted(VALID_SOURCES)}",
                error_code=EnumBindingParseErrorCode.INVALID_SOURCE,
                expression=expression,
            )

        # Parse path segments
        path_segments = tuple(path_str.split("."))

        # Guardrail: no empty segments
        if any(segment == "" for segment in path_segments):
            raise BindingExpressionParseError(
                f"Empty path segment in expression: {expression}",
                error_code=EnumBindingParseErrorCode.EMPTY_PATH_SEGMENT,
                expression=expression,
            )

        # Guardrail: max segments
        if len(path_segments) > effective_max_segments:
            raise BindingExpressionParseError(
                f"Path exceeds max segments ({len(path_segments)} > {effective_max_segments})",
                error_code=EnumBindingParseErrorCode.PATH_TOO_DEEP,
                expression=expression,
            )

        # Validate context paths (allowlist with optional extensions)
        if source == "context":
            # Build effective valid context paths
            effective_context_paths = VALID_CONTEXT_PATHS
            if additional_context_paths:
                effective_context_paths = VALID_CONTEXT_PATHS | additional_context_paths

            if path_segments[0] not in effective_context_paths:
                raise BindingExpressionParseError(
                    f"Invalid context path '{path_segments[0]}'. "
                    f"Must be one of: {sorted(effective_context_paths)}",
                    error_code=EnumBindingParseErrorCode.INVALID_CONTEXT_PATH,
                    expression=expression,
                )

        # Type narrowing: source is guaranteed to be one of the valid values
        return source, path_segments  # type: ignore[return-value]


# =============================================================================
# OperationBindingResolver
# =============================================================================


class OperationBindingResolver:
    """Resolve operation bindings from envelope and context.

    This resolver takes pre-parsed bindings (from contract loading) and
    resolves them against an event envelope and optional context. It handles
    global bindings, operation-specific bindings, defaults, and required
    field validation.

    This class is stateless and thread-safe. A single instance can be
    shared across concurrent requests.

    Resolution Order:
        1. Apply global_bindings first
        2. Apply operation-specific bindings (overwrite globals)
        3. Validate required fields (fail fast on None)
        4. Apply defaults for optional fields
        5. Return resolved parameters

    Example:
        >>> resolver = OperationBindingResolver()
        >>> result = resolver.resolve(
        ...     operation="db.query",
        ...     bindings_subcontract=subcontract,
        ...     envelope=envelope,
        ...     context=context,
        ... )
        >>> if result:
        ...     execute_query(**result.resolved_parameters)

    .. versionadded:: 0.2.6
    """

    def __init__(self) -> None:
        """Initialize resolver with expression parser."""
        self._parser = BindingExpressionParser()

    def resolve(
        self,
        operation: str,
        bindings_subcontract: ModelOperationBindingsSubcontract,
        envelope: object,
        context: object | None,
        correlation_id: UUID | None = None,
    ) -> ModelBindingResolutionResult:
        """Resolve all bindings for an operation.

        Resolution order:
            1. Apply global_bindings first
            2. Apply operation-specific bindings (overwrite globals)
            3. Validate required fields
            4. Return resolved parameters or fail fast

        Args:
            operation: Operation name (e.g., ``"db.query"``).
            bindings_subcontract: Subcontract with binding definitions
                (pre-parsed from contract.yaml).
            envelope: Event envelope with payload. Can be a dict or
                Pydantic model with ``payload`` attribute.
            context: Dispatch context (may be None). Can be a dict or
                Pydantic model.
            correlation_id: Request correlation for error context and
                distributed tracing.

        Returns:
            ``ModelBindingResolutionResult`` with resolved parameters.
            Check ``result.success`` or use ``if result:`` to verify
            resolution succeeded.

        Raises:
            BindingResolutionError: If a required binding resolves to None
                (fail-fast behavior). The error includes diagnostic context
                including operation name, parameter name, and expression.

        Example:
            >>> result = resolver.resolve(
            ...     operation="db.query",
            ...     bindings_subcontract=subcontract,
            ...     envelope={"payload": {"sql": "SELECT 1"}},
            ...     context={"now_iso": "2025-01-01T00:00:00Z"},
            ... )
            >>> result.resolved_parameters
            {'sql': 'SELECT 1', 'timestamp': '2025-01-01T00:00:00Z'}
        """
        resolved_parameters: dict[str, JsonType] = {}
        resolved_from: dict[str, str] = {}

        # Collect all bindings to process (global first, then operation-specific)
        # Operation-specific bindings override globals for the same parameter
        bindings_to_process: list[ModelParsedBinding] = []

        if bindings_subcontract.global_bindings:
            bindings_to_process.extend(bindings_subcontract.global_bindings)

        operation_bindings = bindings_subcontract.bindings.get(operation, [])
        bindings_to_process.extend(operation_bindings)

        # Get max JSON recursion depth from contract (or use default)
        max_json_depth = bindings_subcontract.max_json_recursion_depth

        # Process each binding
        for binding in bindings_to_process:
            try:
                value = self._resolve_single_binding(
                    binding=binding,
                    envelope=envelope,
                    context=context,
                    max_json_recursion_depth=max_json_depth,
                )

                if value is None and binding.required:
                    # Fail fast on missing required binding
                    raise BindingResolutionError(
                        f"Required binding '{binding.parameter_name}' resolved to None",
                        operation_name=operation,
                        parameter_name=binding.parameter_name,
                        expression=binding.original_expression,
                        missing_segment=binding.path_segments[-1]
                        if binding.path_segments
                        else None,
                        correlation_id=correlation_id,
                    )

                # Use default if value is None and not required
                if value is None and binding.default is not None:
                    value = binding.default

                resolved_parameters[binding.parameter_name] = value
                resolved_from[binding.parameter_name] = binding.original_expression

            except BindingResolutionError:
                # Re-raise BindingResolutionError without wrapping
                raise
            except Exception as e:
                # Wrap unexpected errors with diagnostic context
                raise BindingResolutionError(
                    f"Failed to resolve binding: {e}",
                    operation_name=operation,
                    parameter_name=binding.parameter_name,
                    expression=binding.original_expression,
                    correlation_id=correlation_id,
                ) from e

        return ModelBindingResolutionResult(
            operation_name=operation,
            resolved_parameters=resolved_parameters,
            resolved_from=resolved_from,
            success=True,
            error=None,
        )

    def _resolve_single_binding(
        self,
        binding: ModelParsedBinding,
        envelope: object,
        context: object | None,
        max_json_recursion_depth: int = DEFAULT_JSON_RECURSION_DEPTH,
    ) -> JsonType:
        """Resolve a single binding from envelope/context.

        Args:
            binding: Pre-parsed binding with source and path.
            envelope: Event envelope (dict or Pydantic model).
            context: Dispatch context (dict, Pydantic model, or None).
            max_json_recursion_depth: Maximum depth for JSON compatibility
                validation. Defaults to DEFAULT_JSON_RECURSION_DEPTH (100).

        Returns:
            Resolved value (may be None if path doesn't exist).

        .. versionchanged:: 0.2.7
            Added max_json_recursion_depth parameter for per-contract configuration.
        """
        # Get source object based on binding source
        if binding.source == "payload":
            source_obj = self._get_payload(envelope)
        elif binding.source == "envelope":
            source_obj = envelope
        elif binding.source == "context":
            source_obj = context
        else:
            # Should not happen if bindings are pre-validated
            return None

        if source_obj is None:
            return None

        # Traverse path to get value
        return self._traverse_path(
            source_obj,
            binding.path_segments,
            max_json_recursion_depth=max_json_recursion_depth,
        )

    def _get_payload(self, envelope: object) -> object | None:
        """Extract payload from envelope.

        Supports both dict-based envelopes and Pydantic model envelopes.

        Args:
            envelope: Event envelope.

        Returns:
            Payload object or None if not found.
        """
        if isinstance(envelope, dict):
            return envelope.get("payload")
        if hasattr(envelope, "payload"):
            return getattr(envelope, "payload", None)
        return None

    def _traverse_path(
        self,
        obj: object,
        path_segments: tuple[str, ...],
        max_json_recursion_depth: int = DEFAULT_JSON_RECURSION_DEPTH,
    ) -> JsonType:
        """Traverse path segments to get value.

        Uses ``dict.get()`` for dicts and ``getattr()`` for objects.
        Returns None if any segment in the path doesn't exist.

        Runtime Validation:
            The returned value is validated at runtime to ensure JSON
            compatibility. If the traversed value is not JSON-compatible
            (e.g., a callable, custom object, or other non-serializable type),
            a warning is logged and None is returned for graceful degradation.

            This validation prevents non-JSON values from propagating through
            the binding resolution pipeline, which would cause serialization
            failures downstream.

        Args:
            obj: Starting object to traverse from.
            path_segments: Tuple of field names to traverse.
            max_json_recursion_depth: Maximum depth for JSON compatibility
                validation. Defaults to DEFAULT_JSON_RECURSION_DEPTH (100).

        Returns:
            Value at path or None if:
            - Path doesn't exist
            - Value at path is not JSON-compatible (logs warning)

        .. versionchanged:: 0.2.7
            Added max_json_recursion_depth parameter for per-contract configuration.
        """
        current: object = obj

        for segment in path_segments:
            if current is None:
                return None

            if isinstance(current, dict):
                current = current.get(segment)
            elif isinstance(current, BaseModel):
                # Pydantic models: use getattr
                current = getattr(current, segment, None)
            elif hasattr(current, segment):
                # Generic objects with attributes
                current = getattr(current, segment, None)
            else:
                # No way to access the segment
                return None

        # Runtime validation: ensure value is JSON-compatible
        # The type checker sees 'object' but binding paths should only
        # resolve to JSON-compatible values. We validate at runtime to
        # catch cases where non-JSON values (e.g., callables, custom objects)
        # are accidentally exposed through the binding path.
        if not _is_json_compatible(current, max_depth=max_json_recursion_depth):
            logger.warning(
                "Binding path '%s' resolved to non-JSON-compatible value of type '%s'. "
                "Returning None for graceful degradation.",
                ".".join(path_segments),
                type(current).__name__,
            )
            return None

        return current  # type: ignore[return-value]


__all__: list[str] = [
    "BindingExpressionParseError",
    "BindingExpressionParser",
    "EnumBindingParseErrorCode",
    "OperationBindingResolver",
]
