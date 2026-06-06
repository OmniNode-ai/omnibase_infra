# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker subnet-pool exhaustion alert model + threshold builder."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.observability.runner_health.model_network_pool_status import (
    ModelNetworkPoolStatus,
)


class ModelNetworkPoolAlert(BaseModel):
    """Alert payload fired before Docker subnet-pool exhaustion."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Trace correlation ID")
    alert_type: Literal["docker_subnet_pool_pressure"] = Field(
        default="docker_subnet_pool_pressure"
    )
    host: str = Field(..., description="Runner host")
    network_count: int = Field(..., ge=0)
    pool_capacity: int = Field(..., gt=0)
    remaining_capacity: int = Field(..., ge=0)
    reclaim_candidate_count: int = Field(
        default=0,
        ge=0,
        description="Networks the janitor judged reclaimable this pass",
    )

    @model_validator(mode="after")
    def _capacity_invariant(self) -> ModelNetworkPoolAlert:
        if self.network_count > self.pool_capacity + self.remaining_capacity:
            raise ValueError("network_count exceeds pool_capacity + remaining_capacity")
        return self

    def to_slack_message(self) -> str:
        """Format the pool-pressure alert for Slack."""
        lines = [
            f":satellite: *Docker Subnet Pool Pressure* on `{self.host}`",
            "",
            f"Networks: *{self.network_count}/{self.pool_capacity}* "
            f"(remaining: {self.remaining_capacity})",
        ]
        if self.reclaim_candidate_count > 0:
            lines.append(
                f"Janitor can reclaim *{self.reclaim_candidate_count}* "
                "stale owned network(s) this pass."
            )
        else:
            lines.append(
                ":warning: No reclaimable networks — pressure is from active "
                "lanes or unowned networks. Manual review required."
            )
        return "\n".join(lines)


def build_pool_alert_if_pressured(
    status: ModelNetworkPoolStatus,
    correlation_id: UUID,
    reclaim_candidate_count: int = 0,
) -> ModelNetworkPoolAlert | None:
    """Return an alert iff the pool is at/over the pre-exhaustion threshold."""
    if not status.is_over_threshold:
        return None
    return ModelNetworkPoolAlert(
        correlation_id=correlation_id,
        host=status.host,
        network_count=status.network_count,
        pool_capacity=status.pool_capacity,
        remaining_capacity=status.remaining_capacity,
        reclaim_candidate_count=reclaim_candidate_count,
    )


__all__ = ["ModelNetworkPoolAlert", "build_pool_alert_if_pressured"]
