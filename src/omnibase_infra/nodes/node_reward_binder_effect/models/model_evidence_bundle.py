# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Collection of evidence items for an evaluation run.

Stub model pending OMN-2537 merge (canonical models in omnibase_core).

Ticket: OMN-2552
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_reward_binder_effect.models.model_evidence_item import (
    ModelEvidenceItem,
)


class ModelEvidenceBundle(BaseModel):
    """Collection of evidence items for an evaluation run.

    Stub pending OMN-2537 merge into omnibase_core.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_id: UUID = Field(default_factory=uuid4, description="Unique bundle ID.")
    items: tuple[ModelEvidenceItem, ...] = Field(
        default_factory=tuple, description="Evidence items."
    )
    run_id: UUID = Field(..., description="Evaluation run this bundle belongs to.")


__all__: list[str] = ["ModelEvidenceBundle"]
