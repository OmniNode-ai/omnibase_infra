# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Service health check result model for the setup validate effect node.

Ticket: OMN-3491
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSetupNodeHealthResult(BaseModel):
    """Health check result for a single deployment node (service).

    Frozen and immutable — safe for tuple collection in output models.

    Note: Named ``ModelSetupNodeHealthResult`` rather than
    ``ModelServiceHealthResult`` to follow ONEX domain-specific terminology
    (class_anti_pattern rule).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_label: str = Field(
        ...,
        description="Label identifying the deployment node that was health-checked.",
    )
    healthy: bool = Field(
        ...,
        description="Whether the node passed its health check.",
    )
    message: str = Field(
        ...,
        description="Human-readable health check result summary.",
    )
    detail: str | None = Field(
        default=None,
        description="Optional extended detail or error context.",
    )
    response_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Round-trip latency of the health check in milliseconds.",
    )


# Public alias for backward-compatibility with ticket spec naming.
ModelServiceHealthResult = ModelSetupNodeHealthResult

__all__: list[str] = ["ModelSetupNodeHealthResult", "ModelServiceHealthResult"]
