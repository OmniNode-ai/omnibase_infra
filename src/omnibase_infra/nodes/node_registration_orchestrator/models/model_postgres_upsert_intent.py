# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL upsert intent model for registration orchestrator.

This module provides the typed intent model for PostgreSQL upsert operations.
Registered with RegistryIntent for dynamic type resolution.

Related:
    - ModelRegistryIntent: Base class for all registration intents
    - RegistryIntent: Decorator-based registry for intent type discovery
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_intent_payload import (
    ModelPostgresIntentPayload,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_registry_intent import (
    ModelRegistryIntent,
    RegistryIntent,
)


@RegistryIntent.register("postgres")
class ModelPostgresUpsertIntent(ModelRegistryIntent):
    """Intent to upsert node registration in PostgreSQL.

    This model is registered with RegistryIntent for dynamic type resolution,
    enabling Pydantic discriminated union validation without explicit union types.

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
        strict=True,
        from_attributes=True,
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
    payload: ModelPostgresIntentPayload = Field(
        ...,
        description="PostgreSQL-specific registration payload",
    )


__all__ = [
    "ModelPostgresUpsertIntent",
]
