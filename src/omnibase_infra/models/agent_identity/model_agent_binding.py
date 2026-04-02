# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model for agent-to-terminal binding."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelAgentBinding(BaseModel):
    """Represents an agent's current binding to a terminal session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    terminal_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Stable terminal tab identifier",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Current Claude Code session ID",
    )
    machine: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Machine hostname or identifier",
    )
    bound_at: datetime = Field(
        ...,
        description="When this binding was established (UTC, timezone-aware)",
    )
