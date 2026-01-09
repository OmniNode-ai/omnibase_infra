# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul registration intent model for registration orchestrator.

This module provides the typed intent model for Consul registration operations.
Registered with RegistryIntent for dynamic type resolution.

Related:
    - ModelRegistryIntent: Base class for all registration intents
    - RegistryIntent: Decorator-based registry for intent type discovery
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_intent_payload import (
    ModelConsulIntentPayload,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_registry_intent import (
    ModelRegistryIntent,
    RegistryIntent,
)


@RegistryIntent.register("consul")
class ModelConsulRegistrationIntent(ModelRegistryIntent):
    """Intent to register a node in Consul service discovery.

    This model is registered with RegistryIntent for dynamic type resolution,
    enabling Pydantic discriminated union validation without explicit union types.

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
        strict=True,
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
    payload: ModelConsulIntentPayload = Field(
        ...,
        description="Consul-specific registration payload",
    )


__all__ = [
    "ModelConsulRegistrationIntent",
]
