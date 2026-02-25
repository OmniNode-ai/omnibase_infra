# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Single unit of evidence supporting an evaluation.

Stub model pending OMN-2537 merge (canonical models in omnibase_core).

Ticket: OMN-2552
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelEvidenceItem(BaseModel):
    """Single unit of evidence supporting an evaluation.

    Stub pending OMN-2537 merge into omnibase_core.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: UUID = Field(default_factory=uuid4, description="Unique ID for this item.")
    source: str = Field(..., description="Source identifier (e.g. 'session_log').")
    content: str = Field(..., description="Raw evidence content.")
    weight: float = Field(default=1.0, ge=0.0, description="Relative weighting.")


__all__: list[str] = ["ModelEvidenceItem"]
