# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output model for event forwarding."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelEventForwardResult(BaseModel):
    """Result of forwarding an event to an HTTP backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    success: bool = Field(..., description="Whether forwarding succeeded.")
    http_status: int = Field(default=0, description="HTTP status code from backend.")
    endpoint: str = Field(default="", description="Backend endpoint that was called.")
    error_message: str = Field(
        default="", description="Error detail if success is False."
    )
