# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry health response model for API health checks.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelRegistryHealthResponse(BaseModel):
    """Health check response for the registry API.

    Attributes:
        status: Overall health status (healthy, degraded, unhealthy)
        timestamp: When the health check was performed
        components: Health status of individual components
        version: API version string
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall health status",
    )
    timestamp: datetime = Field(
        ...,
        description="When the health check was performed",
    )
    components: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Health status of individual components",
    )
    version: str = Field(
        default="1.0.0",
        description="API version string",
    )


__all__ = ["ModelRegistryHealthResponse"]
