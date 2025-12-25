# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Failed Component Model.

This module provides the Pydantic model for representing a component that
failed during shutdown operations.

Migration Notes:
    This model replaces the legacy tuple[str, str] pattern for failed component
    tracking. The legacy pattern used (component_type, error_message) tuples,
    which lacked type safety and semantic clarity.

    Legacy compatibility methods are provided for gradual migration:
    - from_legacy_tuple(): Convert from tuple[str, str]
    - to_legacy_tuple(): Convert back to tuple[str, str]

    Part of OMN-1007 tuple-to-model conversion work.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelFailedComponent(BaseModel):
    """Represents a component that failed during shutdown.

    Provides a strongly-typed alternative to the legacy tuple[str, str] pattern
    for tracking component failures with associated error messages.

    Attributes:
        component_name: Name or type identifier of the failed component.
        error_message: Error message describing the failure reason.

    Example:
        >>> failed = ModelFailedComponent(
        ...     component_name="KafkaEventBus",
        ...     error_message="Connection timeout during shutdown"
        ... )
        >>> print(failed.component_name)
        KafkaEventBus
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    component_name: str = Field(
        min_length=1,
        description="Name or type identifier of the failed component",
    )
    error_message: str = Field(
        min_length=1,
        description="Error message describing the failure reason",
    )

    @classmethod
    def from_legacy_tuple(cls, t: tuple[str, str]) -> ModelFailedComponent:
        """Create a ModelFailedComponent from a legacy tuple.

        Provides backward compatibility for code that still uses the
        tuple[str, str] pattern.

        Args:
            t: Legacy tuple of (component_name, error_message).

        Returns:
            A new ModelFailedComponent instance.

        Example:
            >>> legacy = ("KafkaEventBus", "Connection failed")
            >>> failed = ModelFailedComponent.from_legacy_tuple(legacy)
            >>> failed.component_name
            'KafkaEventBus'
        """
        return cls(component_name=t[0], error_message=t[1])

    def to_legacy_tuple(self) -> tuple[str, str]:
        """Convert to legacy tuple format.

        Provides backward compatibility for code that expects the
        tuple[str, str] pattern.

        Returns:
            Tuple of (component_name, error_message).

        Example:
            >>> failed = ModelFailedComponent(
            ...     component_name="VaultAdapter",
            ...     error_message="Auth expired"
            ... )
            >>> failed.to_legacy_tuple()
            ('VaultAdapter', 'Auth expired')
        """
        return (self.component_name, self.error_message)

    def __str__(self) -> str:
        """Return a human-readable string representation.

        Returns:
            String format showing component name and error message.
        """
        return f"{self.component_name}: {self.error_message}"


__all__: list[str] = ["ModelFailedComponent"]
