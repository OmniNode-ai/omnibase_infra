# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Output model for Dual Registration Reducer.

This module provides ModelDualRegistrationReducerOutput, the output model for
the pure reducer that emits typed registration intents.

Architecture:
    The dual registration reducer is a PURE reducer that performs no I/O.
    Instead of executing registrations directly, it emits typed intents
    that describe the desired side effects:

    - ModelConsulRegisterIntent: Declares Consul service registration
    - ModelPostgresUpsertRegistrationIntent: Declares PostgreSQL record upsert

    An Effect node receives these intents and executes the actual I/O operations.

Thread Safety:
    ModelDualRegistrationReducerOutput is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Related:
    - OMN-889: Infrastructure MVP - ModelNodeIntrospectionEvent
    - OMN-912: ModelIntent typed payloads
    - docs/handoffs/HANDOFF_PURE_REDUCER_ARCHITECTURE.md: Architecture decision
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from omnibase_core.models.intents import ModelCoreRegistrationIntent
from pydantic import BaseModel, ConfigDict, Field


class ModelDualRegistrationReducerOutput(BaseModel):
    """Output model for the dual registration reducer.

    Contains the typed intents emitted by the pure reducer, along with
    metadata about the reduction operation. The Effect layer is responsible
    for executing these intents and reporting actual success/failure.

    Pure Reducer Semantics:
        This output represents the INTENT to register, not the registration
        result. The reducer cannot know if registration will succeed because
        it performs no I/O. The status field reflects intent emission outcome:

        - "success": Both Consul and PostgreSQL intents were emitted
        - "partial": Only one intent could be emitted (validation failure)
        - "failed": No intents could be emitted (event validation failed)

    Attributes:
        node_id: Unique identifier of the node being registered.
        intents: Tuple of typed registration intents to be executed by Effect.
        status: Intent emission outcome (not registration outcome).
        consul_intent_emitted: Whether Consul registration intent was emitted.
        postgres_intent_emitted: Whether PostgreSQL registration intent was emitted.
        validation_error: Error message if validation failed (None if valid).
        processing_time_ms: Time taken to build intents in milliseconds.
        correlation_id: Request correlation ID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_core.models.intents import ModelConsulRegisterIntent
        >>> output = ModelDualRegistrationReducerOutput(
        ...     node_id=uuid4(),
        ...     intents=(
        ...         ModelConsulRegisterIntent(
        ...             service_id="node-compute-abc123",
        ...             service_name="onex-compute",
        ...             tags=["node_type:compute"],
        ...             correlation_id=uuid4(),
        ...         ),
        ...     ),
        ...     status="partial",
        ...     consul_intent_emitted=True,
        ...     postgres_intent_emitted=False,
        ...     validation_error="PostgreSQL record validation failed",
        ...     processing_time_ms=5.2,
        ...     correlation_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Identity
    node_id: UUID = Field(
        ..., description="Unique identifier of the node being registered"
    )

    # Emitted intents (tuple for immutability)
    intents: tuple[ModelCoreRegistrationIntent, ...] = Field(
        default=(),
        description="Typed registration intents to be executed by Effect layer",
    )

    # Intent emission status (NOT registration status)
    status: Literal["success", "partial", "failed"] = Field(
        ...,
        description=(
            "Intent emission outcome: 'success' = both intents emitted, "
            "'partial' = one intent emitted, 'failed' = no intents emitted"
        ),
    )
    consul_intent_emitted: bool = Field(
        ..., description="Whether Consul registration intent was emitted"
    )
    postgres_intent_emitted: bool = Field(
        ..., description="Whether PostgreSQL registration intent was emitted"
    )

    # Error tracking
    validation_error: str | None = Field(
        default=None, description="Error message if validation failed"
    )

    # Performance metrics
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken to build intents in milliseconds",
    )

    # Tracing
    correlation_id: UUID = Field(
        ..., description="Request correlation ID for distributed tracing"
    )


__all__ = ["ModelDualRegistrationReducerOutput"]
