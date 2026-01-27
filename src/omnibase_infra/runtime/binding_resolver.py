# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Binding expression parser and resolver for operation bindings.

This module provides:

- **BindingExpressionParser**: Parses ${source.path} expressions with guardrails
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

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from omnibase_core.types import JsonType
from omnibase_infra.errors import BindingResolutionError
from omnibase_infra.models.bindings import (
    ModelBindingResolutionResult,
    ModelOperationBindingsSubcontract,
    ModelParsedBinding,
)

if TYPE_CHECKING:
    from uuid import UUID


# =============================================================================
# Guardrail Constants
# =============================================================================

MAX_EXPRESSION_LENGTH: int = 256
"""Maximum allowed length for binding expressions (characters)."""

MAX_PATH_SEGMENTS: int = 20
"""Maximum allowed path depth (dot-separated segments)."""

VALID_SOURCES: frozenset[str] = frozenset({"payload", "envelope", "context"})
"""Valid source names for binding expressions."""

VALID_CONTEXT_PATHS: frozenset[str] = frozenset(
    {
        "now_iso",
        "dispatcher_id",
        "correlation_id",
    }
)
"""Exhaustive list of valid context paths.

Context paths are special runtime-provided values that are injected by the
dispatch infrastructure. Adding new context paths requires updating this
list AND the dispatch context provider.
"""

# Expression pattern: ${source.path.to.field}
# - Group 1: source (lowercase letters only)
# - Group 2: path (letters, numbers, underscores, dots)
EXPRESSION_PATTERN: re.Pattern[str] = re.compile(r"^\$\{([a-z]+)\.([a-zA-Z0-9_.]+)\}$")
"""Compiled regex for parsing binding expressions."""


# =============================================================================
# BindingExpressionParser
# =============================================================================


class BindingExpressionParser:
    """Parse ${source.path} expressions with guardrails.

    This parser validates and decomposes binding expressions into their
    constituent parts: source and path segments. It enforces all guardrails
    at parse time to fail fast on invalid expressions.

    Guardrails enforced:
        - Max expression length: 256 characters
        - Max path depth: 20 segments
        - No array indexing (``[0]``, ``[*]``)
        - Source must be: ``payload`` | ``envelope`` | ``context``
        - Context paths must be in ``VALID_CONTEXT_PATHS``

    This class is stateless and thread-safe.

    Example:
        >>> parser = BindingExpressionParser()
        >>> source, path = parser.parse("${payload.user.id}")
        >>> source
        'payload'
        >>> path
        ('user', 'id')

    .. versionadded:: 0.2.6
    """

    def parse(
        self,
        expression: str,
    ) -> tuple[Literal["payload", "envelope", "context"], tuple[str, ...]]:
        """Parse expression into (source, path_segments).

        Args:
            expression: Expression in ``${source.path.to.field}`` format.

        Returns:
            Tuple of (source, path_segments) where source is one of
            ``"payload"``, ``"envelope"``, ``"context"`` and path_segments
            is a tuple of field names.

        Raises:
            ValueError: If expression is malformed or violates any guardrail:
                - Expression exceeds max length (256 chars)
                - Expression contains array access (``[``, ``]``)
                - Expression syntax doesn't match pattern
                - Source is not valid
                - Path contains empty segments
                - Path exceeds max segments (20)
                - Context path is not in allowlist

        Example:
            >>> parser = BindingExpressionParser()
            >>> parser.parse("${payload.user.email}")
            ('payload', ('user', 'email'))

            >>> parser.parse("${context.now_iso}")
            ('context', ('now_iso',))

            >>> parser.parse("${payload.items[0]}")  # Raises ValueError
            Traceback (most recent call last):
                ...
            ValueError: Array access not allowed in expressions: ${payload.items[0]}
        """
        # Guardrail: max length
        if len(expression) > MAX_EXPRESSION_LENGTH:
            raise ValueError(
                f"Expression exceeds max length ({len(expression)} > {MAX_EXPRESSION_LENGTH})"
            )

        # Guardrail: no array access
        if "[" in expression or "]" in expression:
            raise ValueError(f"Array access not allowed in expressions: {expression}")

        # Parse expression
        match = EXPRESSION_PATTERN.match(expression)
        if not match:
            raise ValueError(
                f"Invalid expression syntax: {expression}. "
                f"Expected format: ${{source.path.to.field}}"
            )

        source = match.group(1)
        path_str = match.group(2)

        # Validate source
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of: {sorted(VALID_SOURCES)}"
            )

        # Parse path segments
        path_segments = tuple(path_str.split("."))

        # Guardrail: no empty segments
        if any(segment == "" for segment in path_segments):
            raise ValueError(f"Empty path segment in expression: {expression}")

        # Guardrail: max segments
        if len(path_segments) > MAX_PATH_SEGMENTS:
            raise ValueError(
                f"Path exceeds max segments ({len(path_segments)} > {MAX_PATH_SEGMENTS})"
            )

        # Validate context paths (exhaustive allowlist)
        if source == "context":
            if path_segments[0] not in VALID_CONTEXT_PATHS:
                raise ValueError(
                    f"Invalid context path '{path_segments[0]}'. "
                    f"Must be one of: {sorted(VALID_CONTEXT_PATHS)}"
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

        # Process each binding
        for binding in bindings_to_process:
            try:
                value = self._resolve_single_binding(
                    binding=binding,
                    envelope=envelope,
                    context=context,
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
    ) -> JsonType:
        """Resolve a single binding from envelope/context.

        Args:
            binding: Pre-parsed binding with source and path.
            envelope: Event envelope (dict or Pydantic model).
            context: Dispatch context (dict, Pydantic model, or None).

        Returns:
            Resolved value (may be None if path doesn't exist).
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
        return self._traverse_path(source_obj, binding.path_segments)

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
    ) -> JsonType:
        """Traverse path segments to get value.

        Uses ``dict.get()`` for dicts and ``getattr()`` for objects.
        Returns None if any segment in the path doesn't exist.

        Args:
            obj: Starting object to traverse from.
            path_segments: Tuple of field names to traverse.

        Returns:
            Value at path or None if path doesn't exist.
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

        # Return the value - type checker sees 'object' but we know
        # it's JSON-compatible due to how binding paths work
        return current  # type: ignore[return-value]


__all__: list[str] = [
    "BindingExpressionParser",
    "MAX_EXPRESSION_LENGTH",
    "MAX_PATH_SEGMENTS",
    "OperationBindingResolver",
    "VALID_CONTEXT_PATHS",
    "VALID_SOURCES",
]
