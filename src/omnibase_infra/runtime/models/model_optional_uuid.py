# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for optional UUID values in runtime module.

This module provides a strongly-typed Pydantic model for optional UUID
values, replacing `UUID | None` union types to comply with ONEX standards.

Design Note - Code Duplication:
    The optional wrapper models (ModelOptionalUUID, ModelOptionalString,
    ModelOptionalCorrelationId) contain similar methods (get(), has_value(),
    get_or_default(), __bool__()). This duplication is intentional to comply
    with ONEX principles:

    1. Each model is self-contained and independently testable
    2. No inheritance or mixin dependencies (composition over inheritance)
    3. Explicit behavior for each type without generic base classes
    4. Each model can evolve independently without breaking others
    5. Clear type signatures without generic type parameters

    While this creates some duplication, it provides stronger type safety,
    better testability, and clearer semantics for each optional type.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelOptionalUUID(BaseModel):
    """Strongly-typed model for optional UUID values.

    Replaces `UUID | None` to comply with ONEX standards requiring specific
    typed models instead of generic union types.

    This wrapper provides a consistent API for working with optional UUIDs,
    including presence checking, default value handling, and functional
    transformation methods.

    Attributes:
        value: The optional UUID value, defaults to None.

    Example:
        >>> from uuid import UUID
        >>> opt = ModelOptionalUUID(value=UUID("12345678-1234-5678-1234-567812345678"))
        >>> opt.has_value()
        True
        >>> empty = ModelOptionalUUID()
        >>> empty.has_value()
        False
    """

    model_config = ConfigDict(frozen=True, from_attributes=True, extra="forbid")

    value: UUID | None = Field(default=None, description="Optional UUID value")

    def get(self) -> UUID | None:
        """Get the optional value.

        Returns:
            The UUID value if present, None otherwise.
        """
        return self.value

    def has_value(self) -> bool:
        """Check if value is present.

        Returns:
            True if value is not None, False otherwise.
        """
        return self.value is not None

    def get_or_default(self, default: UUID) -> UUID:
        """Get value or return default if None.

        Args:
            default: The default UUID to return if None.

        Returns:
            The stored value if present, otherwise the default.
        """
        return self.value if self.value is not None else default

    def __bool__(self) -> bool:
        """Boolean representation based on value presence.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when a value is present. This differs from typical
            Pydantic model behavior where ``bool(model)`` always returns ``True`` for
            any valid model instance.

            This design enables idiomatic presence checks::

                if opt_uuid:
                    # Value is present - use it
                    process(opt_uuid.value)
                else:
                    # No value - use fallback
                    use_default()

            Use ``opt_uuid.has_value()`` for explicit, self-documenting code.
            Use ``opt_uuid is not None`` if you need to check model existence.

        Returns:
            True if value is present, False otherwise.
        """
        return self.has_value()


__all__ = ["ModelOptionalUUID"]
