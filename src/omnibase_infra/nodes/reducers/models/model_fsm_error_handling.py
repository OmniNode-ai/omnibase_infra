# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Error Handling Model.

This module provides the ModelFSMErrorHandling for configuring error handling
behavior in the finite state machine.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.reducers.models.model_fsm_retry_policy import (
    ModelFSMRetryPolicy,
)


class ModelFSMErrorHandling(BaseModel):
    """Error handling configuration for FSM.

    Defines how the FSM handles errors during execution. Supports both
    full structured format (production) and simplified format (test fixtures).

    Attributes:
        default_error_state: The state to transition to on unhandled errors.
        default_action: Simple error action (for simplified format compatibility).
        timeout_seconds: Maximum time in seconds for FSM operations.
        retry_policy: Configuration for retry behavior on failures.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_error_state: str = Field(
        default="registration_failed",
        description="The state to transition to on unhandled errors",
    )
    default_action: str | None = Field(
        default=None,
        description="Simple error action (for simplified format compatibility)",
    )
    timeout_seconds: int = Field(
        default=30, ge=1, description="Maximum time in seconds for FSM operations"
    )
    retry_policy: ModelFSMRetryPolicy = Field(
        default_factory=ModelFSMRetryPolicy,
        description="Configuration for retry behavior on failures",
    )


__all__ = ["ModelFSMErrorHandling"]
