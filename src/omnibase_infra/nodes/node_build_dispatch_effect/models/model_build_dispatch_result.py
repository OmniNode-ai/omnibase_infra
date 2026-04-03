# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelBuildDispatchResult — result from the build dispatch effect node.

Related:
    - OMN-7318: node_build_dispatch_effect
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_outcome import (
    ModelBuildDispatchOutcome,
)


class ModelDelegationPayload(BaseModel):
    """A delegation request payload to be published by the orchestrator.

    Effect handlers must not publish events directly — they return payloads
    for the orchestrator to publish (architectural rule: only orchestrators
    may access the event bus).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str = Field(..., description="Logical event type for routing.")
    topic: str = Field(..., description="Kafka topic to publish to.")
    # NOTE: Any is required because delegation payloads carry arbitrary
    # JSON-serialisable data from various upstream sources (e.g., prompt
    # parameters, task metadata) whose schema is not known at compile time.
    payload: dict[str, Any] = Field(..., description="JSON-serialisable event payload.")
    correlation_id: UUID = Field(..., description="Tracing correlation ID.")


class ModelBuildDispatchResult(BaseModel):
    """Result from the build dispatch effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Build loop cycle correlation ID.")
    outcomes: tuple[ModelBuildDispatchOutcome, ...] = Field(
        ..., description="Per-ticket dispatch outcomes."
    )
    total_dispatched: int = Field(
        default=0, ge=0, description="Successfully dispatched count."
    )
    total_failed: int = Field(default=0, ge=0, description="Failed dispatch count.")
    delegation_payloads: tuple[ModelDelegationPayload, ...] = Field(
        default=(),
        description="Delegation request payloads for the orchestrator to publish.",
    )


__all__: list[str] = ["ModelBuildDispatchResult", "ModelDelegationPayload"]
