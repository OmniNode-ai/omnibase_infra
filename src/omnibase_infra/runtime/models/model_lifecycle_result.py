# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Lifecycle Result Model for handler shutdown operations.

This module provides the ModelLifecycleResult class for representing the outcome
of individual handler lifecycle operations (e.g., shutdown).

Design Pattern:
    ModelLifecycleResult replaces tuple[str, bool, str | None] returns from
    shutdown_handler() with a strongly-typed, self-documenting model that
    provides factory methods for success/failure cases.

    The model uses an empty string sentinel instead of None for error_message
    to eliminate union types and reduce type complexity. Use the `has_error`
    property to check if an error exists before accessing error_message.

Thread Safety:
    ModelLifecycleResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.runtime.models import ModelLifecycleResult
    >>>
    >>> # Create a successful shutdown result
    >>> success = ModelLifecycleResult.succeeded("kafka")
    >>> success.handler_type
    'kafka'
    >>> success.success
    True
    >>> success.has_error
    False
    >>>
    >>> # Create a failed shutdown result
    >>> failure = ModelLifecycleResult.failed("db", "Connection timeout")
    >>> failure.success
    False
    >>> failure.error_message
    'Connection timeout'
    >>> failure.has_error
    True

.. versionchanged:: 0.7.0
    Changed error_message from str | None to str with empty string sentinel
    to reduce union count (OMN-1003 consolidation).
"""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError


class ModelLifecycleResult(BaseModel):
    """Result of a single handler lifecycle operation (e.g., shutdown).

    Provides a strongly-typed replacement for tuple[str, bool, str | None]
    with factory methods for common success/failure patterns.

    Note:
        The ``error_message`` field uses an empty string sentinel instead
        of None to eliminate union types. Use the ``has_error`` property
        to check if an error message exists before accessing it.

    Attributes:
        handler_type: The handler type identifier (e.g., "kafka", "db", "http").
        success: True if the lifecycle operation completed successfully.
        error_message: Error description if the operation failed, empty string
            otherwise. Use ``has_error`` to check if an error exists.

    Example:
        >>> result = ModelLifecycleResult.succeeded("consul")
        >>> result.success
        True
        >>> result.has_error
        False

        >>> result = ModelLifecycleResult.failed("vault", "Auth expired")
        >>> result.success
        False
        >>> result.error_message
        'Auth expired'
        >>> result.has_error
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    handler_type: str = Field(
        ...,
        description="The handler type identifier (e.g., 'kafka', 'db').",
        min_length=1,
    )
    success: bool = Field(
        ...,
        description="True if the lifecycle operation completed successfully.",
    )
    error_message: str = Field(
        default="",
        description="Error description if the operation failed, empty string otherwise. "
        "Use has_error property to check if an error exists.",
    )

    @property
    def has_error(self) -> bool:
        """Check if an error message exists.

        Returns:
            True if error_message is non-empty, False otherwise.

        Example:
            >>> ModelLifecycleResult.succeeded("kafka").has_error
            False
            >>> ModelLifecycleResult.failed("db", "Error").has_error
            True
        """
        return bool(self.error_message)

    @classmethod
    def succeeded(cls, handler_type: str) -> "ModelLifecycleResult":
        """Create a successful lifecycle result.

        Args:
            handler_type: The handler type identifier.

        Returns:
            ModelLifecycleResult indicating success.

        Example:
            >>> result = ModelLifecycleResult.succeeded("kafka")
            >>> result.success
            True
            >>> result.has_error
            False
        """
        return cls(handler_type=handler_type, success=True, error_message="")

    @classmethod
    def failed(cls, handler_type: str, error_message: str) -> "ModelLifecycleResult":
        """Create a failed lifecycle result.

        Args:
            handler_type: The handler type identifier.
            error_message: Description of the error that occurred. Must be non-empty.

        Returns:
            ModelLifecycleResult indicating failure with error details.

        Raises:
            ValueError: If error_message is empty.

        Example:
            >>> result = ModelLifecycleResult.failed("db", "Connection refused")
            >>> result.success
            False
            >>> result.error_message
            'Connection refused'
        """
        if not error_message:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="lifecycle_failed",
            )
            raise ProtocolConfigurationError(
                "error_message must be non-empty for failed results",
                context=context,
            )
        return cls(
            handler_type=handler_type, success=False, error_message=error_message
        )

    def is_success(self) -> bool:
        """Check if the lifecycle operation succeeded.

        Returns:
            True if success is True, False otherwise.
        """
        return self.success

    def is_failure(self) -> bool:
        """Check if the lifecycle operation failed.

        Returns:
            True if success is False, False otherwise.
        """
        return not self.success

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``success`` is True. This differs from typical
            Pydantic model behavior where ``bool(model)`` always returns ``True`` for
            any valid model instance.

            This design enables idiomatic lifecycle operation checks::

                if result:
                    # Operation succeeded - continue
                    continue_shutdown_sequence()
                else:
                    # Operation failed - handle error
                    log_error(result.error_message)
                    raise LifecycleError(result.error_message)

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
            True if operation succeeded, False otherwise.

        Example:
            >>> if ModelLifecycleResult.succeeded("kafka"):
            ...     print("Shutdown complete!")
            Shutdown complete!
        """
        return self.success

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String format: "ModelLifecycleResult(handler_type='...', success=...)"
        """
        status = "succeeded" if self.success else f"failed: {self.error_message}"
        return f"ModelLifecycleResult(handler_type='{self.handler_type}', {status})"


__all__ = ["ModelLifecycleResult"]
