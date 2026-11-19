# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A typed heartbeat health verdict for a running runtime (OMN-24496)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums.enum_runtime_heartbeat_refusal_reason import (
    EnumRuntimeHeartbeatRefusalReason,
)


class ModelRuntimeHeartbeatHealth(BaseModel):
    """One heartbeat health verdict for a runtime whose heartbeat task runs.

    Every runtime heartbeats: a running runtime reports whether its heartbeats
    are succeeding, and an unhealthy verdict carries the refusal reason and the
    count of consecutive failures behind it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    healthy: bool = Field(
        ..., description="Whether the runtime's heartbeats are succeeding"
    )
    reason: EnumRuntimeHeartbeatRefusalReason | None = Field(
        default=None,
        description="Why the heartbeats fail, set only on an unhealthy verdict",
    )
    detail: str = Field(
        default="",
        description="Human-readable detail behind the latest heartbeat failure",
    )
    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="How many consecutive heartbeats have failed so far",
    )
    published_once: bool = Field(
        default=False,
        description="Whether at least one heartbeat has been published",
    )

    @model_validator(mode="after")
    def _reason_matches_health(self) -> ModelRuntimeHeartbeatHealth:
        """Refuse a verdict whose health and reason disagree."""
        if not self.healthy and self.reason is None:
            raise ValueError("an unhealthy verdict must carry a reason")
        if self.healthy and self.reason is not None:
            raise ValueError("a healthy verdict must not carry a reason")
        return self


__all__ = ["ModelRuntimeHeartbeatHealth"]
