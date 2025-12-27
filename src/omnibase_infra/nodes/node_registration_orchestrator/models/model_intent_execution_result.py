# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent execution result model for registration orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelIntentExecutionResult(BaseModel):
    """Result of executing a single intent.

    Captures the outcome of a single registration intent execution,
    including success/failure status, timing, and error details.

    Attributes:
        intent_kind: The type of intent that was executed (e.g., 'consul', 'postgres').
        success: Whether the execution completed successfully.
        error: Error message if execution failed, None otherwise.
        execution_time_ms: Time taken to execute the intent in milliseconds.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        from_attributes=True,
    )

    intent_kind: str = Field(
        ...,
        min_length=1,
        description="The intent kind that was executed",
    )
    success: bool = Field(
        ...,
        description="Whether execution succeeded",
    )
    error: str | None = Field(
        default=None,
        description="Error message if failed",
    )
    execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Execution time in milliseconds",
    )


__all__ = ["ModelIntentExecutionResult"]
