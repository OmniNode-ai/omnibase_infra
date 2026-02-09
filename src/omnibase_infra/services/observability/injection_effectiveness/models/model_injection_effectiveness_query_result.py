# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Paginated query result model for injection effectiveness reads.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.observability.injection_effectiveness.models.model_injection_effectiveness_query import (
    ModelInjectionEffectivenessQuery,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_injection_effectiveness_row import (
    ModelInjectionEffectivenessRow,
)


class ModelInjectionEffectivenessQueryResult(BaseModel):
    """Paginated query result for injection effectiveness data.

    Attributes:
        rows: List of matching rows.
        total_count: Total matching rows (before pagination).
        has_more: Whether more results exist beyond this page.
        query: The original query for reference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: tuple[ModelInjectionEffectivenessRow, ...] = Field(
        default_factory=tuple, description="Matching rows"
    )
    total_count: int = Field(default=0, ge=0, description="Total matching rows")
    has_more: bool = Field(default=False, description="More results available")
    query: ModelInjectionEffectivenessQuery = Field(..., description="Original query")


__all__ = ["ModelInjectionEffectivenessQueryResult"]
