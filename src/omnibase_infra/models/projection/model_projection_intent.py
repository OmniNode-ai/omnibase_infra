# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ModelProjectionIntent — local stub until OMN-2509 (omnibase_core) merges.

OMN-2509 will refactor NodeReducer to emit ModelProjectionIntent as part of
its effects list rather than calling the projector directly.  Until that PR
lands and the new omnibase-core package is published, omnibase_infra stubs
the model here so that:

    - DispatchResultApplier can gate Kafka publish on projection completion
    - Integration tests can exercise the ordering guarantee end-to-end
    - The runtime pipeline is wire-ready the moment OMN-2509 ships

Migration path (OMN-2510 follow-up):
    Once omnibase_core>=0.19.0 exposes ModelProjectionIntent, delete this
    file and update all imports to point at the canonical location.

This model is intentionally minimal — it carries only the fields that the
runtime needs in order to route the intent to NodeProjectionEffect.execute().
"""

from __future__ import annotations

from typing import (
    Any,
    Literal,
)  # NOTE: OMN-2510 - Any used for dynamic projection payload; schema validated by projector
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionIntent(BaseModel):
    """Intent emitted by the reducer to trigger synchronous projection.

    The reducer pipeline produces one ModelProjectionIntent per aggregate
    state change.  The runtime intercepts this intent, hands it to
    NodeProjectionEffect, and waits for completion before publishing any
    Kafka messages.

    Fields:
        intent_type: Routing key — always ``"projection.write"`` so the
            runtime can distinguish projection intents from other intent
            types without an isinstance check.
        subject: The aggregate type being projected (e.g., "NodeRegistration").
        aggregate_id: UUID of the specific aggregate instance.
        projection_type: Name of the projection table / projector class.
        payload: The data to persist (arbitrary mapping, projector validates).
        causation_event_id: UUID of the event that caused this projection.
        correlation_id: Propagated from the originating request for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["projection.write"] = Field(
        default="projection.write",
        description="Routing key for the projection effect handler",
    )
    subject: str = Field(
        ...,
        description="Aggregate type being projected (e.g., 'NodeRegistration')",
    )
    aggregate_id: UUID = Field(
        ...,
        description="UUID of the specific aggregate instance",
    )
    projection_type: str = Field(
        ...,
        description="Name of the projection table or projector class",
    )
    # NOTE: OMN-2510 - Dynamic projection payload; concrete schema is validated
    # by the projector implementation (NodeProjectionEffect), not the intent.
    # Using Any here is intentional: the runtime cannot know the schema of every
    # projector upfront, and enforcing it here would couple the intent model to
    # each projector's data contract.
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Data to persist — projector validates schema",
    )
    causation_event_id: UUID | None = Field(
        default=None,
        description="UUID of the event that caused this projection (for audit trail)",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Propagated correlation ID for distributed tracing",
    )


__all__ = ["ModelProjectionIntent"]
