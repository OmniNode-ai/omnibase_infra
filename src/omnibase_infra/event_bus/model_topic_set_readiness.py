# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Readiness outcome for one contract's topic set (OMN-13237, §3.7).

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_topic_readiness_failure import (
    ModelTopicReadinessFailure,
)


class ModelTopicSetReadiness(BaseModel):
    """Readiness outcome for one contract's topic set.

    Attributes:
        topics: The topic set the readiness confirm was run against.
        status: Aggregate readiness status for the set.
        ready_topics: Topics that converged on broker metadata.
        failures: Per-topic classified failures (empty when status is READY).
        attempts: Number of metadata poll attempts performed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topics: tuple[str, ...] = Field(default_factory=tuple)
    status: EnumTopicReadinessStatus = Field(default=EnumTopicReadinessStatus.SKIPPED)
    ready_topics: tuple[str, ...] = Field(default_factory=tuple)
    failures: tuple[ModelTopicReadinessFailure, ...] = Field(default_factory=tuple)
    attempts: int = Field(default=0, ge=0)

    @property
    def is_ready(self) -> bool:
        """True only when every topic in the set converged."""
        return self.status is EnumTopicReadinessStatus.READY


__all__: list[str] = ["ModelTopicSetReadiness"]
