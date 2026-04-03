# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelDelegationPayload — delegation request for orchestrator publishing.

Effect handlers must not publish events directly — they return payloads
for the orchestrator to publish (architectural rule: only orchestrators
may access the event bus).

Related:
    - OMN-7381: wire Kafka delegation dispatch
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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


__all__: list[str] = ["ModelDelegationPayload"]
