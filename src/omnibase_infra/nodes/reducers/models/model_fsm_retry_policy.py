# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Retry Policy Model.

This module provides the ModelFSMRetryPolicy for configuring retry behavior
in FSM error handling.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelFSMRetryPolicy(BaseModel):
    """Retry policy configuration for FSM error handling.

    Defines how the FSM should retry failed operations.

    Attributes:
        max_retries: Maximum number of retry attempts.
        backoff_type: Type of backoff strategy (e.g., 'exponential', 'linear').
        initial_delay_ms: Initial delay in milliseconds before first retry.
        max_delay_ms: Maximum delay in milliseconds between retries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(
        default=3, ge=0, description="Maximum number of retry attempts"
    )
    backoff_type: str = Field(
        default="exponential",
        description="Type of backoff strategy (e.g., 'exponential', 'linear')",
    )
    initial_delay_ms: int = Field(
        default=1000,
        ge=0,
        description="Initial delay in milliseconds before first retry",
    )
    max_delay_ms: int = Field(
        default=30000, ge=0, description="Maximum delay in milliseconds between retries"
    )


__all__ = ["ModelFSMRetryPolicy"]
