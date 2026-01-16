# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for policy type filter values in runtime module.

This module provides a strongly-typed Pydantic model for policy type
filter values, replacing `str | EnumPolicyType | None` union types
to comply with ONEX standards.
"""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumPolicyType


class ModelPolicyTypeFilter(BaseModel):
    """Strongly-typed model for policy type filter values.

    Replaces `str | EnumPolicyType | None` to comply with ONEX standards
    requiring specific typed models instead of generic union types.

    This model is used in PolicyRegistry filtering operations where the
    policy type can be specified as:
    - A string literal (e.g., "orchestrator", "reducer")
    - An EnumPolicyType enum value
    - None (no filter, match all types)

    The model provides normalization methods to convert any input format
    to a consistent string representation for internal use.

    Attributes:
        string_value: The policy type as a string, if provided as string.
        enum_value: The policy type as an enum, if provided as enum.

    Example:
        >>> filter1 = ModelPolicyTypeFilter(string_value="orchestrator")
        >>> filter1.normalize()
        'orchestrator'
        >>> filter2 = ModelPolicyTypeFilter(enum_value=EnumPolicyType.REDUCER)
        >>> filter2.normalize()
        'reducer'
        >>> filter3 = ModelPolicyTypeFilter()
        >>> filter3.normalize() is None
        True
    """

    string_value: str | None = Field(
        default=None, description="Policy type as string literal"
    )
    enum_value: EnumPolicyType | None = Field(
        default=None, description="Policy type as EnumPolicyType enum"
    )

    model_config = ConfigDict(
        frozen=True,  # Make immutable like ModelPolicyKey
        extra="forbid",
        from_attributes=True,
    )

    @classmethod
    def from_string(cls, value: str) -> "ModelPolicyTypeFilter":
        """Create filter from a string value.

        Args:
            value: The policy type as a string.

        Returns:
            A new ModelPolicyTypeFilter with string_value set.
        """
        return cls(string_value=value)

    @classmethod
    def from_enum(cls, value: EnumPolicyType) -> "ModelPolicyTypeFilter":
        """Create filter from an EnumPolicyType value.

        Args:
            value: The policy type as an enum.

        Returns:
            A new ModelPolicyTypeFilter with enum_value set.
        """
        return cls(enum_value=value)

    @classmethod
    def none(cls) -> "ModelPolicyTypeFilter":
        """Create an empty filter (matches all types).

        Returns:
            A new ModelPolicyTypeFilter with no value set.
        """
        return cls()

    def normalize(self) -> str | None:
        """Normalize the filter value to a string or None.

        Converts the internal representation to a consistent string
        format for use in filtering operations.

        Returns:
            The policy type as a string if set, None otherwise.
            Enum values are converted to their string representation.
        """
        if self.enum_value is not None:
            return self.enum_value.value
        return self.string_value

    def has_value(self) -> bool:
        """Check if a filter value is present.

        Returns:
            True if either string_value or enum_value is set.
        """
        return self.string_value is not None or self.enum_value is not None

    def matches(self, policy_type: str) -> bool:
        """Check if this filter matches a given policy type.

        Args:
            policy_type: The policy type string to check against.

        Returns:
            True if:
            - This filter has no value (matches all types)
            - The normalized filter value equals the given policy type
        """
        normalized = self.normalize()
        if normalized is None:
            return True
        return normalized == policy_type

    def __bool__(self) -> bool:
        """Boolean representation based on value presence.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when a filter value is set. This differs from typical
            Pydantic model behavior where ``bool(model)`` always returns ``True`` for
            any valid model instance.

            This design enables idiomatic filter checks::

                if policy_filter:
                    # Filter is active - apply it
                    filtered = [p for p in policies if policy_filter.matches(p.type)]
                else:
                    # No filter - return all
                    filtered = policies

            Use ``policy_filter.has_value()`` for explicit, self-documenting code.
            Use ``policy_filter is not None`` if you need to check model existence.

        Returns:
            True if a filter value is set, False otherwise.
        """
        return self.has_value()


__all__ = ["ModelPolicyTypeFilter"]
