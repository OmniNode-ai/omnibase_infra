# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A single topic's classified readiness failure (OMN-13237, §3.7).

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.enum_topic_readiness_failure_reason import (
    EnumTopicReadinessFailureReason,
)


class ModelTopicReadinessFailure(BaseModel):
    """A single topic's classified readiness failure.

    Attributes:
        topic: The full ONEX topic name that failed readiness confirm.
        reason: Classified failure reason (§3.7).
        detail: Human-readable detail (no secrets).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str
    reason: EnumTopicReadinessFailureReason
    detail: str = Field(default="")


__all__: list[str] = ["ModelTopicReadinessFailure"]
