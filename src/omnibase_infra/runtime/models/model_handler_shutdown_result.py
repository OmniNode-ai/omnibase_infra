# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Shutdown Result Model.

This module provides the Pydantic model for handler shutdown operation results.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelHandlerShutdownResult(BaseModel):
    """Result of a handler shutdown operation.

    Encapsulates the result of shutting down handlers by priority,
    tracking which handlers succeeded and which failed with their error messages.

    Attributes:
        succeeded_handlers: List of handler types that shutdown successfully.
        failed_handlers: List of (handler_type, error_message) tuples for failures.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    succeeded_handlers: list[str] = Field(
        default_factory=list,
        description="Handler types that shutdown successfully",
    )
    failed_handlers: list[tuple[str, str | None]] = Field(
        default_factory=list,
        description="Handler types that failed with error messages",
    )


__all__: list[str] = ["ModelHandlerShutdownResult"]
