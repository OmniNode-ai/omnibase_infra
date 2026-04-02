# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input model for RSD data fetch effect node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRsdDataFetchRequest(BaseModel):
    """Request to fetch ticket and dependency data for RSD scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    ticket_ids: tuple[str, ...] = Field(
        ..., description="Ticket IDs to fetch data for."
    )
    include_dependencies: bool = Field(
        default=True, description="Whether to fetch dependency graph data."
    )
    include_agent_requests: bool = Field(
        default=True, description="Whether to fetch agent request data."
    )
    include_plan_overrides: bool = Field(
        default=True, description="Whether to fetch plan override data."
    )
