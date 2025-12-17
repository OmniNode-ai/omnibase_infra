# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Capabilities Model.

This module provides ModelNodeCapabilities for strongly-typed node capabilities
in the ONEX 2-way registration pattern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeCapabilities(BaseModel):
    """Strongly-typed node capabilities model.

    Replaces dict[str, Any] with explicit capability fields.
    Uses extra="allow" to support custom capabilities while
    providing type safety for known fields.

    Known capability fields are typed explicitly. Additional custom
    capabilities can be added via the extra="allow" config, and they
    will be stored as model extra fields accessible via model_extra.

    Attributes:
        postgres: Whether node has PostgreSQL capability.
        read: Whether node has read capability.
        write: Whether node has write capability.
        database: Whether node has generic database capability.
        processing: Whether node has processing capability.
        batch_size: Optional batch size limit.
        max_batch: Optional maximum batch size.
        supported_types: List of supported data types.
        routing: Whether node has routing capability.
        config: Nested configuration dictionary.
        transactions: Whether node supports transactions.
        feature: Generic feature flag.

    Example:
        >>> caps = ModelNodeCapabilities(
        ...     postgres=True,
        ...     read=True,
        ...     write=True,
        ... )
        >>> caps.postgres
        True

        >>> # Custom capabilities via extra="allow"
        >>> caps = ModelNodeCapabilities(
        ...     custom_capability=True,  # type: ignore[call-arg]
        ...     another_field="value",  # type: ignore[call-arg]
        ... )
        >>> caps.model_extra["custom_capability"]
        True
    """

    model_config = ConfigDict(
        extra="allow",  # Accept additional fields not explicitly defined
        frozen=False,  # Allow updates (ModelNodeRegistration is mutable)
        from_attributes=True,
    )

    # Database capabilities
    postgres: bool = Field(default=False, description="PostgreSQL capability")
    read: bool = Field(default=False, description="Read capability")
    write: bool = Field(default=False, description="Write capability")
    database: bool = Field(default=False, description="Generic database capability")
    transactions: bool = Field(default=False, description="Transaction support")

    # Processing capabilities
    processing: bool = Field(default=False, description="Processing capability")
    batch_size: int | None = Field(default=None, description="Batch size limit")
    max_batch: int | None = Field(default=None, description="Maximum batch size")
    supported_types: list[str] = Field(
        default_factory=list, description="Supported data types"
    )

    # Network capabilities
    routing: bool = Field(default=False, description="Routing capability")

    # Generic feature flag (used in tests)
    feature: bool = Field(default=False, description="Generic feature flag")

    # Configuration (nested) - using constrained types instead of Any
    config: dict[str, int | str | bool | float] = Field(
        default_factory=dict, description="Nested configuration"
    )

    def __getitem__(
        self, key: str
    ) -> (
        bool
        | int
        | str
        | float
        | list[str]
        | dict[str, int | str | bool | float]
        | None
    ):
        """Support dict-like access for backwards compatibility.

        Args:
            key: The capability key to access.

        Returns:
            The capability value.

        Raises:
            KeyError: If the key is not found in known fields or extras.
        """
        # Check known fields first
        if hasattr(self, key) and key != "model_config":
            value = getattr(self, key)
            # Don't return default False for bool fields if accessed via []
            # unless it was explicitly set
            if key in self.model_fields_set or value is not False:
                return value  # type: ignore[return-value, no-any-return]
        # Check extra fields
        if self.model_extra and key in self.model_extra:
            return self.model_extra[key]  # type: ignore[return-value, no-any-return]
        # For backwards compatibility, check all known fields
        if key in self.model_fields:
            return getattr(self, key)  # type: ignore[return-value, no-any-return]
        raise KeyError(key)

    def get(
        self,
        key: str,
        default: bool
        | int
        | str
        | float
        | list[str]
        | dict[str, int | str | bool | float]
        | None = None,
    ) -> (
        bool
        | int
        | str
        | float
        | list[str]
        | dict[str, int | str | bool | float]
        | None
    ):
        """Support dict-like get() for backwards compatibility.

        Args:
            key: The capability key to access.
            default: Default value if key not found.

        Returns:
            The capability value or default.
        """
        try:
            return self[key]
        except KeyError:
            return default


__all__ = ["ModelNodeCapabilities"]
