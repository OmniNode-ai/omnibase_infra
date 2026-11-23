# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A command the sender routed to a deploy-agent instance (OMN-26485)."""

from __future__ import annotations

from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict


class ModelRoutedCommand(BaseModel):
    """The command's id is its ``correlation_id``, the id the agent's job record carries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    instance: str
    runtime_lane: str
    requested_by: str
    routed_at: AwareDatetime
