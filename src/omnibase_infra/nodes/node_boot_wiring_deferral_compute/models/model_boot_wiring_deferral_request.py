# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The broker's reading and the deferrable contracts it is judged against."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_boot_wiring_deferral_compute.models.model_boot_wiring_candidate import (
    ModelBootWiringCandidate,
)

__all__ = ["ModelBootWiringDeferralRequest"]


class ModelBootWiringDeferralRequest(BaseModel):
    """Deferrable contracts and the topics the broker reports in use."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    candidates: tuple[ModelBootWiringCandidate, ...] = Field(
        default_factory=tuple,
        description="Contracts the kernel may leave unwired (never a priority one).",
    )
    in_use_topics: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "Topics holding a retained record or unreadable by the measure; "
            "an unreadable topic counts as in use so its contract keeps the boot path."
        ),
    )
