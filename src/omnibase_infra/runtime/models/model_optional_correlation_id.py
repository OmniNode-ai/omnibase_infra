# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for optional correlation ID values in runtime module.

This module provides a strongly-typed Pydantic model for optional
correlation ID values (UUIDs), replacing `UUID | None` union types
to comply with ONEX standards.

Correlation IDs are used throughout the runtime for request tracing,
logging, and distributed system observability.

Design Decision (PR #47 Review):
    This model intentionally duplicates patterns from ModelOptionalUUID rather than
    using inheritance or generics. This design choice provides:

    1. **Self-contained model**: No dependencies on base classes or mixins
    2. **Correlation-specific methods**: `get_or_generate()` is specific to correlation IDs
    3. **ONEX compliance**: Follows "composition over inheritance" principle
    4. **Future-ready**: Available for strongly-typed correlation ID handling
       in RuntimeHostProcess and event envelope processing

    While currently not actively used, this model is pre-built for consistent
    correlation ID handling as the codebase evolves. The trade-off of code
    duplication vs. generic abstractions was intentionally chosen for clarity
    and type safety.

    See Also:
        - ModelOptionalUUID: Similar pattern for generic optional UUIDs
        - docs/patterns/correlation_id_tracking.md: Correlation ID best practices
"""

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(frozen=True, from_attributes=True, extra="forbid")

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

    def __bool__(self) -> bool:
        """Boolean representation based on value presence.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when a correlation ID is present. This differs from
            typical Pydantic model behavior where ``bool(model)`` always returns
            ``True`` for any valid model instance.

            This design enables idiomatic presence checks::

                if corr_id:
                    # Correlation ID is present - propagate it
                    context.set_correlation_id(corr_id.value)
                else:
                    # No correlation ID - generate new one
                    corr_id = corr_id.get_or_generate()

            Use ``corr_id.has_value()`` for explicit, self-documenting code.
            Use ``corr_id is not None`` if you need to check model existence.

        Returns:
            True if correlation ID is present, False otherwise.
        """
        return self.has_value()


__all__ = ["ModelOptionalCorrelationId"]
