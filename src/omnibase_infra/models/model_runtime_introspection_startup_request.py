# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request: one startup introspection attempt of the runtime (OMN-1930, OMN-21590)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)


class ModelRuntimeIntrospectionStartupRequest(BaseModel):
    """What one startup introspection attempt needs to know."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Correlation ID for tracing")
    announcements: tuple[ModelNodeAnnouncement, ...] = Field(
        default=(),
        description="Node announcements published in place of the runtime's own event",
    )
    last_introspection_time: float | None = Field(
        default=None,
        description="Monotonic time of the last published introspection, for throttling",
    )


__all__ = ["ModelRuntimeIntrospectionStartupRequest"]
