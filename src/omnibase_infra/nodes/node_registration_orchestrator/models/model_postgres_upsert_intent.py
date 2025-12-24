# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL upsert intent model for registration orchestrator.

This module provides the typed intent model for PostgreSQL upsert operations.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_intent_payload import (
    ModelPostgresIntentPayload,
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


__all__ = [
    "ModelPostgresUpsertIntent",
]
