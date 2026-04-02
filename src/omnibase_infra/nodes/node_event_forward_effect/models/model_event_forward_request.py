# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input model for event forwarding."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelEventForwardRequest(BaseModel):
    """A platform event to forward to an HTTP backend.

    Consolidates service-lifecycle, system-alert, and tool-update event
    categories from the archive handlers into a single typed model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        default_factory=uuid4, description="Workflow correlation ID."
    )
    event_type: str = Field(..., description="Original event type string.")
    category: Literal["lifecycle", "system", "tool", "generic"] = Field(
        default="generic", description="Event category for backend routing."
    )
    timestamp: str = Field(default="", description="ISO-8601 event timestamp.")
    source: str = Field(default="", description="Originating service or node name.")
    severity: str = Field(
        default="info", description="Event severity: info, warning, error, critical."
    )
    payload: dict[str, object] = Field(  # ONEX_EXCLUDE: dict_str_any
        default_factory=dict,
    )
    metadata: dict[str, object] = Field(  # ONEX_EXCLUDE: dict_str_any
        default_factory=dict,
    )
