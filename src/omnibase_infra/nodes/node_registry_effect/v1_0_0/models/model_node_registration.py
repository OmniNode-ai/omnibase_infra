# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node registration model for registry operations."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeRegistration(BaseModel):
    """Node registration record from storage."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: str
    node_type: str
    node_version: str = "1.0.0"
    capabilities: dict[str, object] = Field(default_factory=dict)
    endpoints: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)
    health_endpoint: str | None = None
    last_heartbeat: datetime | None = None
    registered_at: datetime
    updated_at: datetime
