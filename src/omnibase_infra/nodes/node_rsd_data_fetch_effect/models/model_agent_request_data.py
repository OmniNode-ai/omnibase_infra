# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Agent request data model for RSD scoring."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelAgentRequestData(BaseModel):
    """Fetched agent request for a ticket."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    agent_id: str = Field(min_length=1)  # pattern-ok: agent IDs are opaque strings
    ticket_id: str = Field(min_length=1)  # pattern-ok: Linear ticket IDs are strings
    request_type: str = Field(default="work_request", description="Request type.")
    priority_boost: float = Field(default=0.0, description="Priority boost amount.")
    timestamp: datetime | None = Field(
        default=None, description="When request was made."
    )
    is_active: bool = Field(default=True, description="Whether request is active.")
