# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Persistent agent identity model.

An AgentEntity is the core platform object — a named identity that persists
across sessions, terminals, machines, and channels. It is NOT a session,
NOT a worker, NOT a process. Sessions bind to agents. Workers execute on
behalf of agents. Trust, persona, skills, and memory are properties of agents.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.agent_identity.enum_agent_status import EnumAgentStatus
from omnibase_infra.models.agent_identity.model_agent_binding import ModelAgentBinding


class ModelAgentEntity(BaseModel):
    """A persistent agent identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*$",
        description="Unique agent identifier (e.g., CAIA, SENTINEL, ARCHIVIST)",
    )
    display_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable display name",
    )
    status: EnumAgentStatus = Field(
        default=EnumAgentStatus.IDLE,
        description="Current lifecycle status",
    )
    current_binding: ModelAgentBinding | None = Field(
        default=None,
        description="Current terminal/session binding, or None if unbound",
    )
    active_tickets: tuple[str, ...] = Field(
        default=(),
        description="Ticket IDs currently being worked on",
    )
    current_branch: str | None = Field(
        default=None,
        max_length=256,
        description="Current git branch (from most recent session)",
    )
    working_directory: str | None = Field(
        default=None,
        max_length=1024,
        description="Current working directory (from most recent session)",
    )
    trust_profile_id: str | None = Field(
        default=None,
        description="Reference to TrustProfile (Phase 6)",
    )
    persona_profile_id: str | None = Field(
        default=None,
        description="Reference to PersonaProfile (Phase 3)",
    )
    skill_inventory_id: str | None = Field(
        default=None,
        description="Reference to SkillInventory",
    )
    created_at: datetime = Field(
        ...,
        description="When this agent was created (UTC, timezone-aware)",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Last modification time (for sync ordering and stale detection)",
    )
    revision: int = Field(
        default=1,
        ge=1,
        description="Monotonic revision counter (incremented on every mutation, for conflict resolution)",
    )
    last_active_at: datetime | None = Field(
        default=None,
        description="When this agent was last active (from most recent session end)",
    )
