# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler routing declaration from contract YAML (OMN-7654)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)


class ModelHandlerRouting(BaseModel):
    """Handler routing declaration from contract YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    routing_strategy: str = Field(
        ..., description="Routing strategy (payload_type_match or operation_match)"
    )
    handlers: tuple[ModelHandlerRoutingEntry, ...] = Field(
        default_factory=tuple,
        description="Handler entries",
    )
