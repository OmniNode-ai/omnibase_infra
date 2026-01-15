# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Failed Component Model.

This module provides the Pydantic model for representing a component that
failed during shutdown operations.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelFailedComponent(BaseModel):
    """Represents a component that failed during shutdown.

    Attributes:
        component_name: Name or type identifier of the failed component.
        error_message: Error message describing the failure reason.

    Example:
        >>> failed = ModelFailedComponent(
        ...     component_name="EventBusKafka",
        ...     error_message="Connection timeout during shutdown"
        ... )
        >>> print(failed.component_name)
        EventBusKafka
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

    def __str__(self) -> str:
        """Return a human-readable string representation.

        Returns:
            String format showing component name and error message.
        """
        return f"{self.component_name}: {self.error_message}"


__all__: list[str] = ["ModelFailedComponent"]
