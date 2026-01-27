# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Binding resolution result model.

This model represents the outcome of resolving operation bindings, containing
the resolved parameters and debug information for troubleshooting.

.. versionadded:: 0.2.6
    Created as part of OMN-1518 - Declarative operation bindings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_core.types import JsonType


class ModelBindingResolutionResult(BaseModel):
    """Result of binding resolution - execution fact, never mutated.

    Contains resolved parameters plus debug information for troubleshooting.
    Once created, this is immutable - bindings become execution facts.

    Attributes:
        operation_name: Operation that was resolved.
        resolved_parameters: Mapping of parameter name to resolved value.
        resolved_from: Mapping of parameter name to original expression (debug gold).
        success: True if all required bindings resolved successfully.
        error: Error message if success=False.

    Example:
        >>> result = resolver.resolve(envelope, "db.query")
        >>> if result:
        ...     execute_query(result.resolved_parameters)
        ... else:
        ...     log_error(result.error)

    .. versionadded:: 0.2.6
    """

    operation_name: str = Field(
        ...,
        description="Operation that was resolved",
    )
    resolved_parameters: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Parameter name -> resolved value",
    )
    resolved_from: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter name -> original expression (debug gold)",
    )
    success: bool = Field(
        ...,
        description="True if all required bindings resolved successfully",
    )
    error: str | None = Field(
        default=None,
        description="Error message if success=False",
    )

    model_config = {"frozen": True, "extra": "forbid"}

    def __bool__(self) -> bool:
        """Return True if resolution succeeded.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``success`` is True. This differs from typical
            Pydantic model behavior where ``bool(model)`` always returns ``True`` for
            any valid model instance.

            This design enables idiomatic conditional checks for resolution results::

                result = resolver.resolve(envelope, "db.query")
                if result:
                    # Resolution succeeded - use parameters
                    execute_query(result.resolved_parameters)
                else:
                    # Resolution failed - handle error
                    log_error(result.error)

            If you need to check model existence instead, use explicit attribute access::

                # Check for resolution success (uses __bool__)
                if result:
                    ...

                # Check model is not None
                if result is not None:
                    ...

                # Explicit success check (preferred for clarity)
                if result.success:
                    ...

        Returns:
            True if resolution succeeded, False otherwise.

        .. versionadded:: 0.2.6
        """
        return self.success
