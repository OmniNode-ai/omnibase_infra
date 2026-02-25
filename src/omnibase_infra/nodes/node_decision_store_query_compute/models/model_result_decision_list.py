# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result model for NodeDecisionStoreQueryCompute.

Defines the paginated result returned by cursor-based decision store queries,
including next_cursor for page continuation and total_active for count display.

Note on ModelDecisionStoreEntry dependency:
    ModelDecisionStoreEntry is defined in omnibase_core (OMN-2763, PR #562),
    which is not yet released. The decisions field uses Any at runtime and is
    typed as tuple[ModelDecisionStoreEntry, ...] for static analysis only.
    Once OMN-2763 is merged, this model can be tightened to use the concrete type.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelResultDecisionList(BaseModel):
    """Paginated result of a decision store query.

    Attributes:
        decisions: Ordered tuple of matching decision entries for this page,
            sorted by (created_at DESC, decision_id DESC).
            Elements are ModelDecisionStoreEntry instances at runtime once
            OMN-2763 (omnibase_core) is merged.
        next_cursor: Opaque base64-encoded cursor for retrieving the next page.
            None when there are no further results (last page or empty result).
        total_active: Total count of decisions matching the same WHERE clause,
            computed without LIMIT. Useful for pagination UI ("N of M results").

    Notes:
        - next_cursor is None when len(decisions) < limit (last or only page).
        - total_active reflects the same filters as the query, not just this page.
        - decisions is an empty tuple when no results match the filter.
        - decisions uses Any element type pending OMN-2763 merge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    # NOTE: Using Any pending OMN-2763 (ModelDecisionStoreEntry not yet in installed omnibase_core)
    decisions: tuple[Any, ...] = Field(
        default=(),
        description=(
            "Ordered tuple of matching ModelDecisionStoreEntry instances for this page, "
            "sorted by (created_at DESC, decision_id DESC). "
            "Element type is Any pending OMN-2763 (omnibase_core) merge."
        ),
    )
    next_cursor: str | None = Field(
        default=None,
        description=(
            "Opaque base64-encoded cursor for the next page. "
            "None when there are no further results."
        ),
    )
    total_active: int = Field(
        ...,
        ge=0,
        description=(
            "Total count of decisions matching the same WHERE clause "
            "(not paginated). Used for displaying '1-50 of N' in UIs."
        ),
    )


__all__: list[str] = ["ModelResultDecisionList"]
