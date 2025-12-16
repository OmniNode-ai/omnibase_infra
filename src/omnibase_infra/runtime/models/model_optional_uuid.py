# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for optional UUID values in runtime module.

This module provides a strongly-typed Pydantic model for optional UUID
values, replacing `UUID | None` union types to comply with ONEX standards.
"""

from collections.abc import Callable
from typing import Optional
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

    model_config = ConfigDict(frozen=True)

    value: Optional[UUID] = Field(default=None, description="Optional UUID value")

    def get(self) -> Optional[UUID]:
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

    def map(self, func: Callable[[UUID], UUID]) -> "ModelOptionalUUID":
        """Apply function to value if present.

        Args:
            func: Function to apply to the UUID value.

        Returns:
            A new ModelOptionalUUID with the transformed value,
            or self if value is None.
        """
        if self.value is not None:
            return ModelOptionalUUID(value=func(self.value))
        return self

    def __bool__(self) -> bool:
        """Boolean representation based on value presence.

        Returns:
            True if value is present, False otherwise.
        """
        return self.has_value()


__all__ = ["ModelOptionalUUID"]
