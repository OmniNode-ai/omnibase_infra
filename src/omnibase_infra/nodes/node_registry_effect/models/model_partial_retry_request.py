# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Partial Retry Request Model for targeted backend retries.

This module provides ModelPartialRetryRequest, a Pydantic model that implements
the ProtocolPartialRetryRequest protocol for use with HandlerPartialRetry.

Architecture:
    ModelPartialRetryRequest provides a concrete implementation of the
    ProtocolPartialRetryRequest protocol, allowing the NodeRegistryEffect to
    create properly typed requests for partial failure retries.

Related:
    - HandlerPartialRetry: Handler that consumes this request
    - ProtocolPartialRetryRequest: Protocol this model implements
    - NodeRegistryEffect: Node that creates these requests
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.enums import EnumBackendType


class ModelPartialRetryRequest(BaseModel):
    """Request model for partial failure retry operations.

    Provides a Pydantic model implementation of the ProtocolPartialRetryRequest
    protocol for HandlerPartialRetry.

    Attributes:
        node_id: Unique identifier for the node being registered.
        node_type: Type of ONEX node (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        target_backend: Backend to retry ("consul" or "postgres").
        idempotency_key: Optional key for idempotent retry semantics.
        service_name: Optional service name for Consul registration.
        tags: Tags for Consul service discovery.
        health_check_config: Optional Consul health check configuration.
        endpoints: Dict of endpoint type to URL for PostgreSQL.
        metadata: Additional metadata for PostgreSQL registration.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: UUID = Field(
        ...,
        description="Unique identifier for the node being registered",
    )
    node_type: EnumNodeKind = Field(
        ...,
        description="Type of ONEX node (effect, compute, reducer, orchestrator)",
    )
    node_version: str = Field(
        ...,
        description="Semantic version of the node",
    )
    target_backend: EnumBackendType = Field(
        ...,
        description="Backend to retry: 'consul' or 'postgres'",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Optional key for idempotent retry semantics",
    )
    service_name: str | None = Field(
        default=None,
        description="Name for service discovery registration (Consul)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags for service discovery (Consul)",
    )
    health_check_config: dict[str, str] | None = Field(
        default=None,
        description="Optional health check configuration (Consul)",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Dict of endpoint type to URL (PostgreSQL)",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for the registration record (PostgreSQL)",
    )


__all__ = ["ModelPartialRetryRequest"]
