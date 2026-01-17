# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Field Change Model for Structural Diffs.

Represents a single field change between two snapshots, capturing the
before and after values for diff analysis.

Thread Safety:
    This model is frozen (immutable) for safe sharing across threads.

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelFieldChange(BaseModel):
    """Represents a single field change between snapshots.

    Captures the transition from one value to another for a specific field
    in a structural diff. Both values must be JSON-compatible.

    Attributes:
        from_value: The original value before the change.
        to_value: The new value after the change.

    Example:
        >>> change = ModelFieldChange(
        ...     from_value="pending",
        ...     to_value="active",
        ... )
        >>> change.from_value
        'pending'
        >>> change.to_value
        'active'

        >>> # Complex nested changes
        >>> change = ModelFieldChange(
        ...     from_value={"count": 1, "items": ["a"]},
        ...     to_value={"count": 2, "items": ["a", "b"]},
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    from_value: JsonType = Field(
        ...,
        description="The original value before the change",
    )
    to_value: JsonType = Field(
        ...,
        description="The new value after the change",
    )


__all__: list[str] = ["ModelFieldChange"]
