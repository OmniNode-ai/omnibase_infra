# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Update Model for Storage Operations.

This module provides ModelRegistrationUpdate, representing a partial update
to a registration record. Only fields that are set (not None) will be updated.

Architecture:
    ModelRegistrationUpdate allows selective field updates:
    - endpoints: Optional new endpoints dict
    - metadata: Optional new metadata dict
    - capabilities: Optional new capabilities list
    - node_version: Optional new semantic version (ModelSemVer)

    This model enables type-safe partial updates rather than untyped dict[str, object].

Related:
    - ModelRegistrationRecord: Full record type being updated
    - ProtocolRegistrationStorageHandler: Protocol that uses this for updates
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.models.primitives.model_semver import ModelSemVer


class ModelRegistrationUpdate(BaseModel):
    """Update model for registration storage operations.

    Defines the fields that can be updated in a registration record.
    All fields are optional - only non-None fields will be applied.

    Immutability:
        This model uses frozen=True to ensure updates are immutable
        once created, enabling safe reuse and logging.

    Attributes:
        endpoints: Optional new endpoints dict (replaces existing).
        metadata: Optional new metadata dict (replaces existing).
        capabilities: Optional new capabilities list (replaces existing).
        node_version: Optional new semantic version (replaces existing).

    Example:
        >>> # Update only endpoints
        >>> update = ModelRegistrationUpdate(
        ...     endpoints={"health": "http://new-host:8080/health"},
        ... )

        >>> # Update multiple fields
        >>> update = ModelRegistrationUpdate(
        ...     endpoints={"health": "http://new-host:8080/health"},
        ...     metadata={"team": "platform", "env": "production"},
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    endpoints: dict[str, str] | None = Field(
        default=None,
        description="Optional new endpoints dict (replaces existing if set)",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional new metadata dict (replaces existing if set)",
    )
    capabilities: list[str] | None = Field(
        default=None,
        description="Optional new capabilities list (replaces existing if set)",
    )
    node_version: ModelSemVer | None = Field(
        default=None,
        description="Optional new semantic version",
    )

    @field_validator("node_version", mode="before")
    @classmethod
    def parse_node_version(cls, v: ModelSemVer | str | None) -> ModelSemVer | None:
        """Parse node_version from string or ModelSemVer.

        Args:
            v: Either a ModelSemVer instance, a semver string, or None.

        Returns:
            Validated ModelSemVer instance or None.

        Raises:
            ValueError: If the string is not a valid semantic version.
        """
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return ModelSemVer.parse(v)
            except Exception as e:
                raise ValueError(f"node_version: {e!s}") from e
        return v

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> ModelRegistrationUpdate:
        """Validate that at least one field is set for update.

        An empty update (all None fields) is a no-op and likely a bug.

        Returns:
            The validated update model.

        Raises:
            ValueError: If no fields are set.
        """
        if (
            self.endpoints is None
            and self.metadata is None
            and self.capabilities is None
            and self.node_version is None
        ):
            raise ValueError(
                "At least one field must be set for update. "
                "All fields are currently None."
            )
        return self

    def has_updates(self) -> bool:
        """Check if any fields are set for update.

        Returns:
            True if at least one field is non-None.
        """
        return (
            self.endpoints is not None
            or self.metadata is not None
            or self.capabilities is not None
            or self.node_version is not None
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dict with only non-None fields.

        Returns:
            Dict of field names to values, excluding None fields.
        """
        result: dict[str, object] = {}
        if self.endpoints is not None:
            result["endpoints"] = self.endpoints
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.capabilities is not None:
            result["capabilities"] = self.capabilities
        if self.node_version is not None:
            result["node_version"] = self.node_version
        return result


__all__ = ["ModelRegistrationUpdate"]
