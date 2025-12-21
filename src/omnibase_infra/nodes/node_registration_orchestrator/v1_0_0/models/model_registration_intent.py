# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration intent model for the registration orchestrator."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistrationIntent(BaseModel):
    """Base model for registration intents.

    Intents are typed instructions that the reducer produces and the
    effect node executes. Each intent represents a single infrastructure
    operation to perform.

    Attributes:
        kind: Discriminator for the intent type (e.g., 'consul', 'postgres').
        operation: The specific operation within that kind.
        node_id: Target node ID for the operation.
        correlation_id: Correlation ID for distributed tracing.
        payload: Operation-specific data as a dictionary.

    Note:
        This is a placeholder for the discriminated union of intents
        that will be defined in omnibase_core (OMN-912). The actual
        implementation will use a tagged union pattern like:
        ConsulRegisterIntent | ConsulDeregisterIntent | PostgresUpsertIntent
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    kind: str = Field(
        ...,
        min_length=1,
        description="Intent type discriminator (e.g., 'consul', 'postgres')",
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Operation to perform (e.g., 'register', 'upsert')",
    )
    node_id: UUID = Field(
        ...,
        description="Target node ID for the operation",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    payload: dict[str, object] = Field(
        default_factory=dict,
        description="Operation-specific data (JSON-like structure, typed in OMN-912)",
    )
