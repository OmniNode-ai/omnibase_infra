# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka event payload for LLM endpoint health probe results.

.. versionadded:: 0.9.0
    Part of OMN-2255 LLM endpoint health checker.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.model_llm_endpoint_status import (
    ModelLlmEndpointStatus,
)


class ModelLlmEndpointHealthEvent(BaseModel):
    """Payload emitted as a Kafka event after each probe cycle.

    Attributes:
        timestamp: When the probe cycle completed.
        endpoints: List of per-endpoint status snapshots.
        correlation_id: Trace correlation ID.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    timestamp: datetime = Field(..., description="Probe cycle completion time")
    endpoints: tuple[ModelLlmEndpointStatus, ...] = Field(
        default_factory=tuple,
        description="Per-endpoint status snapshots",
    )
    correlation_id: UUID = Field(..., description="Correlation ID for tracing")


__all__: list[str] = ["ModelLlmEndpointHealthEvent"]
