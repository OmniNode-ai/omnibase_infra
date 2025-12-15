# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for optional correlation ID values in runtime module.

This module provides a strongly-typed Pydantic model for optional
correlation ID values (UUIDs), replacing `UUID | None` union types
to comply with ONEX standards.

Correlation IDs are used throughout the runtime for request tracing,
logging, and distributed system observability.
"""

from collections.abc import Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelOptionalCorrelationId(BaseModel):
    """Strongly-typed model for optional correlation ID values.

    Replaces `UUID | None` for correlation IDs to comply with ONEX
    standards requiring specific typed models instead of generic union types.

    This specialized wrapper is designed for correlation ID use cases,
    providing factory methods to generate new UUIDs when needed and
    ensuring consistent handling across the runtime module.

    Attributes:
        value: The optional correlation ID (UUID), defaults to None.

    Example:
        >>> corr = ModelOptionalCorrelationId.generate()
        >>> corr.has_value()
        True
        >>> empty = ModelOptionalCorrelationId()
        >>> filled = empty.get_or_generate()
        >>> filled.has_value()
        True
    """

    value: UUID | None = Field(
        default=None, description="Optional correlation ID (UUID)"
    )

    @classmethod
    def generate(cls) -> "ModelOptionalCorrelationId":
        """Create a new correlation ID with a generated UUID.

        Factory method that creates a ModelOptionalCorrelationId with
        a newly generated UUID4 value.

        Returns:
            A new ModelOptionalCorrelationId with a generated UUID.
        """
        return cls(value=uuid4())

    @classmethod
    def from_uuid(cls, value: UUID) -> "ModelOptionalCorrelationId":
        """Create a correlation ID from an existing UUID.

        Args:
            value: An existing UUID to wrap.

        Returns:
            A new ModelOptionalCorrelationId with the given UUID.
        """
        return cls(value=value)

    @classmethod
    def none(cls) -> "ModelOptionalCorrelationId":
        """Create an empty correlation ID.

        Returns:
            A new ModelOptionalCorrelationId with no value.
        """
        return cls()

    def get(self) -> UUID | None:
        """Get the optional correlation ID.

        Returns:
            The UUID value if present, None otherwise.
        """
        return self.value

    def set(self, value: UUID | None) -> None:
        """Set the correlation ID.

        Args:
            value: The new UUID value, or None to clear.
        """
        self.value = value

    def has_value(self) -> bool:
        """Check if correlation ID is present.

        Returns:
            True if value is not None, False otherwise.
        """
        return self.value is not None

    def get_or_default(self, default: UUID) -> UUID:
        """Get correlation ID or return default if None.

        Args:
            default: The default UUID to return if None.

        Returns:
            The stored value if present, otherwise the default.
        """
        return self.value if self.value is not None else default

    def get_or_generate(self) -> "ModelOptionalCorrelationId":
        """Get this correlation ID or generate a new one if None.

        Returns a new ModelOptionalCorrelationId with either:
        - The existing UUID if present
        - A newly generated UUID if None

        Returns:
            A ModelOptionalCorrelationId that always has a value.
        """
        if self.value is not None:
            return self
        return ModelOptionalCorrelationId.generate()

    def map(self, func: Callable[[UUID], UUID]) -> "ModelOptionalCorrelationId":
        """Apply function to correlation ID if present.

        Args:
            func: Function to apply to the UUID value.

        Returns:
            A new ModelOptionalCorrelationId with the transformed value,
            or self if value is None.
        """
        if self.value is not None:
            return ModelOptionalCorrelationId(value=func(self.value))
        return self

    def __bool__(self) -> bool:
        """Boolean representation based on value presence.

        Returns:
            True if correlation ID is present, False otherwise.
        """
        return self.has_value()


__all__ = ["ModelOptionalCorrelationId"]
