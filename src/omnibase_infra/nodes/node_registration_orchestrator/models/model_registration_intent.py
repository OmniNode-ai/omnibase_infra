# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration intent models for the registration orchestrator.

This module provides strongly-typed intent models for the registration
orchestrator's infrastructure operations. Each intent type has a specific
payload model, ensuring type safety across the registration workflow.

Design Note:
    Rather than using a loose dict[str, JsonValue] for payloads, we use
    typed payload models that match the exact structure expected by each
    infrastructure adapter. This follows the ONEX principle of "no Any types"
    and provides compile-time validation of intent payloads.

    The pattern uses Literal discriminators for the `kind` field, enabling
    type narrowing in effect node handlers.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.types.json_types import JsonValue


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


class ModelPostgresIntentPayload(BaseModel):
    """Payload for PostgreSQL registration intents.

    Contains the full node introspection data to upsert into the
    registration database. This is a typed representation of the
    data previously passed via model_dump().

    Attributes:
        node_id: Unique node identifier.
        node_type: ONEX node type (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        capabilities: Node capabilities dictionary.
        endpoints: Exposed endpoints (name -> URL).
        node_role: Optional role descriptor.
        metadata: Additional node metadata as JSON-serializable dict.
        network_id: Network/cluster identifier.
        deployment_id: Deployment/release identifier.
        epoch: Registration epoch for ordering.
        timestamp: Event timestamp as ISO string.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    node_id: UUID = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="ONEX node type")
    node_version: str = Field(default="1.0.0", description="Semantic version")
    capabilities: dict[str, JsonValue] = Field(
        default_factory=dict, description="Node capabilities"
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Exposed endpoints"
    )
    node_role: str | None = Field(default=None, description="Node role")
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict, description="Additional metadata"
    )
    correlation_id: UUID = Field(..., description="Correlation ID for tracing")
    network_id: str | None = Field(default=None, description="Network identifier")
    deployment_id: str | None = Field(default=None, description="Deployment identifier")
    epoch: int | None = Field(default=None, ge=0, description="Registration epoch")
    timestamp: str = Field(..., description="Event timestamp as ISO string")


# Type alias for intent payloads
IntentPayload = ModelConsulIntentPayload | ModelPostgresIntentPayload


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


class ModelPostgresUpsertIntent(BaseModel):
    """Intent to upsert node registration in PostgreSQL.

    Attributes:
        kind: Literal discriminator, always "postgres".
        operation: The operation type (e.g., "upsert", "delete").
        node_id: Target node ID for the operation.
        correlation_id: Correlation ID for distributed tracing.
        payload: PostgreSQL-specific registration payload.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    kind: Literal["postgres"] = Field(
        default="postgres",
        description="Intent type discriminator",
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Operation to perform (e.g., 'upsert', 'delete')",
    )
    node_id: UUID = Field(
        ...,
        description="Target node ID for the operation",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    payload: ModelPostgresIntentPayload = Field(
        ...,
        description="PostgreSQL-specific registration payload",
    )


# Discriminated union of all intent types using Annotated pattern
# This enables type narrowing based on the `kind` field
ModelRegistrationIntent = Annotated[
    ModelConsulRegistrationIntent | ModelPostgresUpsertIntent,
    Field(discriminator="kind"),
]


__all__ = [
    "ModelConsulIntentPayload",
    "ModelPostgresIntentPayload",
    "ModelConsulRegistrationIntent",
    "ModelPostgresUpsertIntent",
    "ModelRegistrationIntent",
    "IntentPayload",
]
