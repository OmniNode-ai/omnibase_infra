# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ticket data model for RSD scoring."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelTicketData(BaseModel):
    """Fetched ticket data for RSD scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(min_length=1)  # pattern-ok: Linear ticket IDs are strings
    title: str = Field(default="", description="Ticket title.")
    description: str = Field(default="", description="Ticket description.")
    status: str = Field(default="open")  # pattern-ok: free-text status from Linear API
    priority: str = Field(default="medium", description="Ticket priority label.")
    created_at: datetime | None = Field(default=None, description="Creation timestamp.")
    updated_at: datetime | None = Field(
        default=None, description="Last update timestamp."
    )
    tags: tuple[str, ...] = Field(default_factory=tuple, description="Ticket tags.")
    depends_on: tuple[str, ...] = Field(
        default_factory=tuple, description="Dependency IDs."
    )
    blocks: tuple[str, ...] = Field(
        default_factory=tuple, description="Blocked ticket IDs."
    )
