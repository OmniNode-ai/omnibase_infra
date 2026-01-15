# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shutdown Batch Result Model.

This module provides the Pydantic model for batch shutdown operation results.

Migration Notes:
    The failed_components field was migrated from list[tuple[str, str]] to
    list[ModelFailedComponent] as part of OMN-1007 tuple-to-model conversion.
    This provides better type safety and semantic clarity for failure tracking.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_failed_component import ModelFailedComponent


class ModelShutdownBatchResult(BaseModel):
    """Result of a batch shutdown operation.

    Encapsulates the result of shutting down components by priority,
    tracking which components succeeded and which failed with their error messages.

    Attributes:
        succeeded_components: List of component types that shutdown successfully.
        failed_components: List of ModelFailedComponent instances for failures.

    Example:
        >>> from omnibase_infra.runtime.models import (
        ...     ModelShutdownBatchResult,
        ...     ModelFailedComponent,
        ... )
        >>> result = ModelShutdownBatchResult(
        ...     succeeded_components=["ConsulAdapter", "VaultAdapter"],
        ...     failed_components=[
        ...         ModelFailedComponent(
        ...             component_name="EventBusKafka",
        ...             error_message="Connection timeout"
        ...         )
        ...     ]
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    succeeded_components: list[str] = Field(
        default_factory=list,
        description="Component types that shutdown successfully",
    )
    failed_components: list[ModelFailedComponent] = Field(
        default_factory=list,
        description="Components that failed during shutdown with error details",
    )

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String format showing succeeded and failed counts.
        """
        return (
            f"ModelShutdownBatchResult("
            f"succeeded={len(self.succeeded_components)}, "
            f"failed={len(self.failed_components)})"
        )


__all__: list[str] = ["ModelShutdownBatchResult"]
