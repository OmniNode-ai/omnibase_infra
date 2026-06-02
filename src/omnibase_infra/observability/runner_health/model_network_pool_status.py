# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker subnet-pool occupancy status model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelNetworkPoolStatus(BaseModel):
    """Per-host Docker subnet-pool occupancy snapshot.

    Docker's default address pool subnets a fixed set of CIDR blocks into a
    bounded number of per-network subnets. Once that pool is fully subnetted,
    ``docker network create`` (and therefore every compose-based CI job) fails
    with ``all predefined address pools have been fully subnetted``. Tracking
    ``network_count`` against ``pool_capacity`` lets us alert *before* the pool
    is exhausted instead of discovering it through a red CI run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    host: str = Field(..., description="Runner host inspected")
    network_count: int = Field(
        ..., ge=0, description="Total Docker networks currently on the host"
    )
    pool_capacity: int = Field(
        ...,
        gt=0,
        description="Max networks the configured address pool can subnet",
    )
    warn_threshold_ratio: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Fraction of capacity at which to alert before exhaustion",
    )

    @property
    def utilization_ratio(self) -> float:
        """Fraction of the address pool currently consumed."""
        return self.network_count / self.pool_capacity

    @property
    def remaining_capacity(self) -> int:
        """Networks that can still be created before exhaustion (never < 0)."""
        return max(self.pool_capacity - self.network_count, 0)

    @property
    def is_over_threshold(self) -> bool:
        """True when occupancy has reached the pre-exhaustion alert threshold."""
        return self.network_count >= int(self.pool_capacity * self.warn_threshold_ratio)


__all__ = ["ModelNetworkPoolStatus"]
