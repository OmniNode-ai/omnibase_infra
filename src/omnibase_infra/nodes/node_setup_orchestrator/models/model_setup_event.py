# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Setup event model for the setup orchestrator.

Ticket: OMN-3491
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSetupEvent(BaseModel):
    """A single setup lifecycle event emitted by the orchestrator.

    Frozen and immutable. ``extra=forbid`` enforces the strict contract
    that only declared fields are permitted — no ad-hoc payload injection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_type: str = Field(
        ...,
        description="Event type string; must be one of SETUP_EVENT_TYPES.",
    )
    payload: dict[str, object] = Field(
        default_factory=dict,
        description="Optional structured payload for the event.",
    )


__all__: list[str] = ["ModelSetupEvent"]
