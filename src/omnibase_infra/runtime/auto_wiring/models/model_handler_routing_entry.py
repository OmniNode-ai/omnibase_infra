# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single handler routing entry from contract YAML (OMN-7654)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.models.model_handler_ref import ModelHandlerRef


class ModelHandlerRoutingEntry(BaseModel):
    """A single handler entry from contract handler_routing.handlers[]."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    handler: ModelHandlerRef = Field(..., description="Handler class reference")
    event_model: ModelHandlerRef | None = Field(
        default=None,
        description="Event model reference (payload_type_match strategy)",
    )
    operation: str | None = Field(
        default=None,
        description="Operation name (operation_match strategy)",
    )
