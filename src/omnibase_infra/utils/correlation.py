# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Correlation ID utilities for distributed tracing.

Provides utilities for generating and propagating correlation IDs
across infrastructure components for distributed tracing and debugging.

Correlation IDs are used to:
    - Track requests across service boundaries
    - Link related log entries for debugging
    - Enable distributed tracing in containerized environments
    - Provide unique identifiers for error context

Example:
    >>> from omnibase_infra.utils.correlation import generate_correlation_id
    >>> correlation_id = generate_correlation_id()
    >>> logger.info("Processing request", extra={"correlation_id": correlation_id})

Context Manager Example:
    >>> from omnibase_infra.utils.correlation import CorrelationContext
    >>> with CorrelationContext() as correlation_id:
    ...     logger.info("Operation started", extra={"correlation_id": correlation_id})
    ...     # All operations within this context share the same correlation_id
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from types import TracebackType

# Context variable for correlation ID propagation across async boundaries
# This enables correlation ID to be implicitly passed through async call chains
_correlation_id: ContextVar[UUID | None] = ContextVar("correlation_id", default=None)


def generate_correlation_id() -> UUID:
    """Generate a new correlation ID using UUID4.

    Creates a cryptographically random UUID suitable for distributed tracing.
    UUID4 is used because it provides high uniqueness guarantees without
    requiring coordination between nodes.

    Returns:
        UUID: A new randomly generated UUID4 correlation ID.

    Example:
        >>> correlation_id = generate_correlation_id()
        >>> print(correlation_id)
        123e4567-e89b-12d3-a456-426614174000
        >>> isinstance(correlation_id, UUID)
        True
    """
    return uuid4()


def get_correlation_id() -> UUID:
    """Get current correlation ID from context or generate new one.

    Retrieves the correlation ID from the current async context. If no
    correlation ID has been set in the current context, generates a new
    one and sets it for future retrieval.

    This function is useful when you need a correlation ID but don't know
    if one has already been established in the current execution context.

    Returns:
        UUID: Current correlation ID from context, or a new one if not set.

    Example:
        >>> # First call generates and stores a new ID
        >>> id1 = get_correlation_id()
        >>> # Second call returns the same ID
        >>> id2 = get_correlation_id()
        >>> id1 == id2
        True
    """
    correlation_id = _correlation_id.get()
    if correlation_id is None:
        correlation_id = generate_correlation_id()
        _correlation_id.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: UUID) -> None:
    """Set correlation ID in current context.

    Explicitly sets a correlation ID in the current async context. This is
    useful when you want to propagate an existing correlation ID (e.g., from
    an incoming request header) into the current execution context.

    Args:
        correlation_id: The correlation ID to set. Must be a UUID instance.

    Example:
        >>> incoming_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        >>> set_correlation_id(incoming_id)
        >>> get_correlation_id() == incoming_id
        True
    """
    _correlation_id.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context.

    Resets the correlation ID context variable to None. This is useful
    when starting a new logical operation that should not inherit the
    correlation ID from a previous operation.

    Example:
        >>> set_correlation_id(generate_correlation_id())
        >>> clear_correlation_id()
        >>> _correlation_id.get() is None
        True
    """
    _correlation_id.set(None)


class CorrelationContext:
    """Context manager for correlation ID scoping.

    Provides a context manager that automatically generates or uses a provided
    correlation ID for the duration of the context. The previous correlation ID
    is restored when exiting the context, enabling nested correlation scopes.

    This is useful for:
        - Scoping correlation IDs to specific operations
        - Ensuring cleanup of correlation context
        - Nested correlation contexts with proper restoration

    Attributes:
        correlation_id: The correlation ID used within this context.

    Example:
        >>> with CorrelationContext() as correlation_id:
        ...     logger.info("Operation", extra={"correlation_id": correlation_id})
        ...     # correlation_id is available and set in context

    Example with provided ID:
        >>> existing_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        >>> with CorrelationContext(correlation_id=existing_id) as cid:
        ...     cid == existing_id
        True

    Example with nesting:
        >>> with CorrelationContext() as outer_id:
        ...     with CorrelationContext() as inner_id:
        ...         # inner_id is different from outer_id
        ...         pass
        ...     # outer_id is restored after inner context exits
    """

    def __init__(self, correlation_id: UUID | None = None) -> None:
        """Initialize the correlation context.

        Args:
            correlation_id: Optional correlation ID to use. If not provided,
                a new UUID4 will be generated when entering the context.
        """
        self._correlation_id = correlation_id or generate_correlation_id()
        self._token: Token[UUID | None] | None = None

    @property
    def correlation_id(self) -> UUID:
        """Return the correlation ID for this context.

        Returns:
            UUID: The correlation ID associated with this context.
        """
        return self._correlation_id

    def __enter__(self) -> UUID:
        """Enter the correlation context and set the correlation ID.

        Returns:
            UUID: The correlation ID for this context.
        """
        self._token = _correlation_id.set(self._correlation_id)
        return self._correlation_id

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the correlation context and restore the previous correlation ID.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Traceback if an exception was raised.
        """
        if self._token is not None:
            _correlation_id.reset(self._token)


__all__: list[str] = [
    "CorrelationContext",
    "clear_correlation_id",
    "generate_correlation_id",
    "get_correlation_id",
    "set_correlation_id",
]
