# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Metadata Model.

This module provides ModelNodeMetadata for strongly-typed node metadata
in the ONEX 2-way registration pattern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeMetadata(BaseModel):
    """Strongly-typed node metadata model.

    Replaces dict[str, Any] with explicit metadata fields.
    Uses extra="allow" to support custom metadata while
    providing type safety for known fields.

    Known metadata fields are typed explicitly. Additional custom
    metadata can be added via the extra="allow" config, and they
    will be stored as model extra fields accessible via model_extra.

    Attributes:
        version: Software version string.
        environment: Deployment environment (e.g., production, staging).
        region: Geographic region identifier.
        cluster: Cluster identifier.
        description: Human-readable description (supports Unicode).
        priority: Priority level for scheduling/routing.
        key: Generic key field (used in tests).
        key1: Generic key field (used in tests).
        meta: Generic meta field (used in tests).

    Example:
        >>> meta = ModelNodeMetadata(
        ...     version="1.0.0",
        ...     environment="production",
        ...     region="us-west-2",
        ... )
        >>> meta.environment
        'production'

        >>> # Unicode support
        >>> meta = ModelNodeMetadata(description="Узел обработки")
        >>> meta.description
        'Узел обработки'

        >>> # Custom metadata via extra="allow"
        >>> meta = ModelNodeMetadata(
        ...     custom_field="value",  # type: ignore[call-arg]
        ... )
        >>> meta.model_extra["custom_field"]
        'value'
    """

    model_config = ConfigDict(
        extra="allow",  # Accept additional fields not explicitly defined
        frozen=False,  # Allow updates (ModelNodeRegistration is mutable)
        from_attributes=True,
    )

    # Common metadata fields
    version: str | None = Field(default=None, description="Software version string")
    environment: str | None = Field(
        default=None, description="Deployment environment (production, staging, etc.)"
    )
    region: str | None = Field(default=None, description="Geographic region identifier")
    cluster: str | None = Field(default=None, description="Cluster identifier")
    description: str | None = Field(
        default=None, description="Human-readable description (supports Unicode)"
    )

    # Scheduling/routing metadata
    priority: int | None = Field(
        default=None, description="Priority level for scheduling/routing"
    )

    # Generic fields used in tests - using constrained types
    key: str | None = Field(default=None, description="Generic key field")
    key1: str | None = Field(default=None, description="Generic key1 field")
    meta: str | None = Field(default=None, description="Generic meta field")

    def __getitem__(self, key: str) -> str | int | None:
        """Support dict-like access for backwards compatibility.

        Args:
            key: The metadata key to access.

        Returns:
            The metadata value.

        Raises:
            KeyError: If the key is not found in known fields or extras.
        """
        # Check known fields first
        if hasattr(self, key) and key != "model_config":
            value = getattr(self, key)
            if value is not None:
                return value  # type: ignore[return-value, no-any-return]
        # Check extra fields
        if self.model_extra and key in self.model_extra:
            return self.model_extra[key]  # type: ignore[return-value, no-any-return]
        # For backwards compatibility, check all known fields including None values
        if key in self.model_fields:
            return getattr(self, key)  # type: ignore[return-value, no-any-return]
        raise KeyError(key)

    def get(
        self,
        key: str,
        default: str | int | None = None,
    ) -> str | int | None:
        """Support dict-like get() for backwards compatibility.

        Args:
            key: The metadata key to access.
            default: Default value if key not found.

        Returns:
            The metadata value or default.
        """
        try:
            return self[key]
        except KeyError:
            return default


__all__ = ["ModelNodeMetadata"]
