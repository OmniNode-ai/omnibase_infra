# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for one bounded Docker network janitor pass."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.enum_network_disposition import (
    EnumNetworkDisposition,
)
from omnibase_infra.observability.runner_health.model_network_decision import (
    ModelNetworkDecision,
)


class ModelNetworkJanitorResult(BaseModel):
    """Outcome of one janitor pass over a runner host's networks."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Trace correlation ID")
    ran_at: datetime = Field(..., description="When the janitor pass executed")
    host: str = Field(..., description="Runner host inspected")
    dry_run: bool = Field(..., description="When True, no networks were removed")
    decisions: tuple[ModelNetworkDecision, ...] = Field(
        default_factory=tuple, description="Per-network decision"
    )
    reclaimed: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Network refs actually removed (empty in dry_run)",
    )
    reclaim_errors: tuple[str, ...] = Field(
        default_factory=tuple, description="Per-network removal errors, if any"
    )

    @property
    def reclaim_candidate_count(self) -> int:
        """Networks the janitor judged safe to reclaim this pass."""
        return sum(
            1 for d in self.decisions if d.disposition is EnumNetworkDisposition.RECLAIM
        )

    @property
    def preserved_count(self) -> int:
        """Networks preserved (everything not reclaim-eligible)."""
        return sum(
            1
            for d in self.decisions
            if d.disposition is not EnumNetworkDisposition.RECLAIM
        )


__all__ = ["ModelNetworkJanitorResult"]
