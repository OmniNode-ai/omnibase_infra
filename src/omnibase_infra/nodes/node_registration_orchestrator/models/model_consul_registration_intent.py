# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul registration intent model for registration orchestrator.

This module provides the typed intent model for Consul registration operations.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_intent_payload import (
    ModelConsulIntentPayload,
)


class ModelConsulRegistrationIntent(BaseModel):
    """Intent to register a node in Consul service discovery.

    Attributes:
        kind: Literal discriminator, always "consul".
        operation: The operation type (e.g., "register", "deregister").
        node_id: Target node ID for the operation.
        correlation_id: Correlation ID for distributed tracing.
        payload: Consul-specific registration payload.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    kind: Literal["consul"] = Field(
        default="consul",
        description="Intent type discriminator",
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Operation to perform (e.g., 'register', 'deregister')",
    )
    node_id: UUID = Field(
        ...,
        description="Target node ID for the operation",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    payload: ModelConsulIntentPayload = Field(
        ...,
        description="Consul-specific registration payload",
    )


__all__ = [
    "ModelConsulRegistrationIntent",
]
