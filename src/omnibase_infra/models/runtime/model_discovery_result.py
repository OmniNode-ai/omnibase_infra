# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Discovery Result Model for Handler Discovery Operations.

This module provides the ModelDiscoveryResult model for aggregating the results
of handler discovery operations, including counts and any errors or warnings.

See Also:
    - ModelDiscoveryError: Individual discovery errors
    - ModelDiscoveryWarning: Non-fatal warnings

.. versionadded:: 0.7.0
    Created as part of OMN-1133 contract-based handler discovery.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.runtime.model_discovery_error import ModelDiscoveryError
from omnibase_infra.models.runtime.model_discovery_warning import ModelDiscoveryWarning


class ModelDiscoveryResult(BaseModel):
    """Result of a handler discovery operation.

    Aggregates the results of a discovery operation including counts of
    discovered and registered handlers, any errors or warnings encountered,
    and the timestamp of completion.

    Attributes:
        handlers_discovered: Number of handlers found during discovery.
        handlers_registered: Number of handlers successfully registered.
        errors: List of errors encountered during discovery.
        warnings: List of non-fatal warnings encountered during discovery.
        discovered_at: Timestamp when discovery completed.

    Example:
        >>> result = ModelDiscoveryResult(
        ...     handlers_discovered=5,
        ...     handlers_registered=4,
        ...     errors=[ModelDiscoveryError(
        ...         error_code="MODULE_NOT_FOUND",
        ...         message="Module not found",
        ...     )],
        ... )
        >>> result.has_errors
        True
        >>> bool(result)  # False because there are errors
        False

    .. versionadded:: 0.7.0
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    handlers_discovered: int = Field(
        default=0,
        ge=0,
        description="Number of handlers found during discovery",
    )
    handlers_registered: int = Field(
        default=0,
        ge=0,
        description="Number of handlers successfully registered",
    )
    errors: list[ModelDiscoveryError] = Field(
        default_factory=list,
        description="List of errors encountered during discovery",
    )
    warnings: list[ModelDiscoveryWarning] = Field(
        default_factory=list,
        description="List of non-fatal warnings encountered during discovery",
    )
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when discovery completed",
    )

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during discovery.

        Returns:
            True if errors list is non-empty, False otherwise.

        Example:
            >>> ModelDiscoveryResult().has_errors
            False

        .. versionadded:: 0.7.0
        """
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings occurred during discovery.

        Returns:
            True if warnings list is non-empty, False otherwise.

        Example:
            >>> ModelDiscoveryResult().has_warnings
            False

        .. versionadded:: 0.7.0
        """
        return len(self.warnings) > 0

    def __bool__(self) -> bool:
        """Allow using result in boolean context for success checking.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when no errors occurred during discovery. This differs
            from typical Pydantic model behavior where ``bool(model)`` always returns
            ``True`` for any valid model instance.

            This design enables idiomatic conditional checks for discovery results::

                result = loader.discover_handlers(path)
                if result:
                    # Discovery succeeded - use handlers
                    process_handlers(result.handlers_registered)
                else:
                    # Discovery had errors - handle failures
                    for error in result.errors:
                        log_error(error)

            If you need to check model validity instead, use explicit attribute access::

                # Check for discovery success (uses __bool__)
                if result:
                    ...

                # Check model exists (always True for constructed instance)
                if result is not None:
                    ...

                # Explicit error check (preferred for clarity)
                if not result.has_errors:
                    ...

        Returns:
            True if discovery was successful (no errors), False otherwise.

        Example:
            >>> if ModelDiscoveryResult(handlers_discovered=5, handlers_registered=5):
            ...     print("Success!")
            Success!

        .. versionadded:: 0.7.0
        """
        return len(self.errors) == 0


__all__ = ["ModelDiscoveryResult"]
