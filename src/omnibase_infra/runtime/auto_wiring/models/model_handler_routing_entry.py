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
    event_type: str | None = Field(
        default=None,
        description=(
            "Optional contract-declared event_type alias (e.g., "
            "'omnimarket.pr-lifecycle-orchestrator-start'). When present, the "
            "dispatcher is indexed under this wire-level string in addition to "
            "event_model.name, so publishers that set ModelEventEnvelope.event_type "
            "to the dot-path string resolve to the handler without needing the "
            "Python class name on the wire (OMN-9215)."
        ),
    )
    message_category: str | None = Field(
        default=None,
        description=(
            "Optional per-handler message category override from contract YAML "
            "(EVENT, COMMAND, or INTENT). Required for mixed-topic contracts so "
            "command handlers do not inherit the category of the contract's first "
            "subscribed topic."
        ),
    )
