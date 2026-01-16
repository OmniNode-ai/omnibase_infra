# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for optional string values in runtime module.

This module provides a strongly-typed Pydantic model for optional string
values, replacing `str | None` union types to comply with ONEX standards.
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelOptionalString(BaseModel):
    """Strongly-typed model for optional string values.

    Replaces `str | None` to comply with ONEX standards requiring specific
    typed models instead of generic union types.

    This wrapper provides a consistent API for working with optional strings,
    including presence checking, default value handling, and functional
    transformation methods.

    Attributes:
        value: The optional string value, defaults to None.

    Example:
        >>> opt = ModelOptionalString(value="hello")
        >>> opt.has_value()
        True
        >>> opt.get_or_default("default")
        'hello'
        >>> empty = ModelOptionalString()
        >>> empty.get_or_default("default")
        'default'
    """

    model_config = ConfigDict(frozen=True, from_attributes=True, extra="forbid")

    value: str | None = Field(default=None, description="Optional string value")

    def get(self) -> str | None:
        """Get the optional value.

        Returns:
            The string value if present, None otherwise.
        """
        return self.value

    def has_value(self) -> bool:
        """Check if value is present.

        Returns:
            True if value is not None, False otherwise.
        """
        return self.value is not None

    def get_or_default(self, default: str) -> str:
        """Get value or return default if None.

        Args:
            default: The default value to return if None.

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

                if opt_string:
                    # Value is present - use it
                    process(opt_string.value)
                else:
                    # No value - use fallback
                    use_default()

            Use ``opt_string.has_value()`` for explicit, self-documenting code.
            Use ``opt_string is not None`` if you need to check model existence.

        Returns:
            True if value is present, False otherwise.
        """
        return self.has_value()


__all__ = ["ModelOptionalString"]
