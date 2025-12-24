# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul intent payload model for registration orchestrator.

This module provides the typed payload model for Consul registration intents.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelConsulIntentPayload(BaseModel):
    """Payload for Consul registration intents.

    Used by the Consul adapter to register nodes in service discovery.

    Attributes:
        service_name: Name to register the node as in Consul.
        tags: Optional service tags for Consul filtering.
        meta: Optional metadata key-value pairs.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    service_name: str = Field(
        ...,
        min_length=1,
        description="Service name to register in Consul",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional service tags for Consul filtering",
    )
    meta: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata key-value pairs",
    )


__all__ = [
    "ModelConsulIntentPayload",
]
