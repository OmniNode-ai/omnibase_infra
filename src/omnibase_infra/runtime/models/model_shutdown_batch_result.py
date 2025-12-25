# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shutdown Batch Result Model.

This module provides the Pydantic model for batch shutdown operation results.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelShutdownBatchResult(BaseModel):
    """Result of a batch shutdown operation.

    Encapsulates the result of shutting down components by priority,
    tracking which components succeeded and which failed with their error messages.

    Attributes:
        succeeded_components: List of component types that shutdown successfully.
        failed_components: List of (component_type, error_message) tuples for failures.
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
    failed_components: list[tuple[str, str | None]] = Field(
        default_factory=list,
        description="Component types that failed with error messages",
    )


__all__: list[str] = ["ModelShutdownBatchResult"]
