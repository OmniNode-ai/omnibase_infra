# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration reference parse result model.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.

This module provides the ModelConfigRefParseResult for representing
the result of parsing a configuration reference string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from omnibase_infra.runtime.models.model_config_ref import ModelConfigRef


class ModelConfigRefParseResult(BaseModel):
    """Result of parsing a configuration reference.

    This model follows the result pattern where parsing never throws exceptions
    for invalid input. Instead, check `success` or use the model in a boolean
    context to determine if parsing succeeded.

    Attributes:
        success: True if parsing succeeded, False otherwise.
        config_ref: The parsed ModelConfigRef if successful, None otherwise.
        error_message: Error description if parsing failed, None otherwise.

    Examples:
        >>> result = ModelConfigRef.parse("file:config.yaml")
        >>> if result:
        ...     print(f"Path: {result.config_ref.path}")
        Path: config.yaml
        >>>
        >>> result = ModelConfigRef.parse("invalid")
        >>> if not result:
        ...     print(f"Error: {result.error_message}")
        Error: Invalid config reference format: missing scheme separator ':' in 'invalid'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    success: bool = Field(
        ...,
        description="True if parsing succeeded, False otherwise.",
    )
    # NOTE: Using string annotation to avoid circular import
    config_ref: ModelConfigRef | None = Field(
        default=None,
        description="The parsed configuration reference if successful, None otherwise.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error description if parsing failed, None otherwise.",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``success`` is True. This differs from typical
            Pydantic model behavior where ``bool(model)`` always returns ``True`` for
            any valid model instance.

            This design enables idiomatic parse result checks::

                result = ModelConfigRef.parse("file:config.yaml")
                if result:
                    # Parsing succeeded - use the config_ref
                    use_config(result.config_ref)
                else:
                    # Parsing failed - handle the error
                    log_error(result.error_message)

            If you need to check model validity instead, use explicit attribute access::

                # Check for success (uses __bool__)
                if result:
                    ...

                # Check model is valid (always True for constructed instance)
                if result is not None:
                    ...

                # Explicit success check (preferred for clarity)
                if result.success:
                    ...

        Returns:
            True if parsing succeeded, False otherwise.

        Examples:
            >>> result = ModelConfigRef.parse("file:config.yaml")
            >>> bool(result)
            True
            >>>
            >>> result = ModelConfigRef.parse("")
            >>> bool(result)
            False
        """
        return self.success

    def __str__(self) -> str:
        """Return a human-readable string representation.

        Returns:
            String format showing success status and config_ref or error.
        """
        if self.success:
            return f"ModelConfigRefParseResult(success=True, uri='{self.config_ref.to_uri() if self.config_ref else ''}')"
        return f"ModelConfigRefParseResult(success=False, error='{self.error_message}')"


__all__: list[str] = ["ModelConfigRefParseResult"]
