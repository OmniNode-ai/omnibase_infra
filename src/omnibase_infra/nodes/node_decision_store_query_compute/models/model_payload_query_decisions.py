# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Query payload model for NodeDecisionStoreQueryCompute.

Defines the filter parameters for cursor-paginated decision store queries,
including scope service filtering with ANY/ALL/EXACT mode semantics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Lifecycle statuses for decision_store entries (mirrors DB CHECK constraint).
_DecisionStatus = Literal["PROPOSED", "ACTIVE", "SUPERSEDED", "DEPRECATED"]


class ModelPayloadQueryDecisions(BaseModel):
    """Filter parameters for querying decisions from the decision store.

    Supports cursor-based pagination and flexible scope filtering.
    All filter fields are optional; omitted fields are not applied.

    Attributes:
        domain: Filter by scope_domain (exact match).
        layer: Filter by scope_layer (exact match).
        decision_type: Filter by one or more decision_type values (OR semantics).
        tags: Filter decisions that have ALL of these tags.
        epic_id: Filter by epic_id (exact match).
        status: Filter by lifecycle status. Defaults to "ACTIVE".
        scope_services: List of service slugs to filter by.
            Empty list or None means "do not filter by services".
        scope_services_mode: How scope_services filter is applied.
            ANY — entry overlaps with filter list (at least one match).
            ALL — entry contains all filter services (superset).
            EXACT — entry's scope_services exactly equals filter set.
            Ignored when scope_services is None or empty.
        cursor: Opaque pagination cursor from a previous result's
            next_cursor field. Encodes base64(created_at_iso|decision_id).
        limit: Maximum number of results to return. Defaults to 50.

    Notes:
        - scope_services=None or scope_services=[] returns all decisions
          including platform-wide ones (empty scope_services in DB).
        - cursor ordering is: created_at DESC, decision_id DESC.
        - next_cursor is None when there are no further results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str | None = Field(
        default=None,
        description="Filter by scope_domain (exact match). None = no filter.",
    )
    layer: str | None = Field(
        default=None,
        description=(
            "Filter by scope_layer (exact match). "
            "One of: architecture, design, planning, implementation. "
            "None = no filter."
        ),
    )
    decision_type: list[str] | None = Field(
        default=None,
        description=(
            "Filter by decision_type values (OR semantics). None = no filter."
        ),
    )
    tags: list[str] | None = Field(
        default=None,
        description=("Filter decisions that have ALL of these tags. None = no filter."),
    )
    epic_id: str | None = Field(
        default=None,
        description="Filter by epic_id (exact match). None = no filter.",
    )
    status: _DecisionStatus = Field(
        default="ACTIVE",
        description=(
            "Filter by lifecycle status. "
            "One of: PROPOSED, ACTIVE, SUPERSEDED, DEPRECATED."
        ),
    )
    scope_services: list[str] | None = Field(
        default=None,
        description=(
            "List of service slugs to filter by scope. "
            "None or empty list = do not filter by services "
            "(returns all, including platform-wide decisions)."
        ),
    )
    scope_services_mode: Literal["ANY", "ALL", "EXACT"] = Field(
        default="ANY",
        description=(
            "How scope_services filter is applied when scope_services is non-empty. "
            "ANY: entry scope_services overlaps with filter list (JSONB ?| operator). "
            "ALL: entry scope_services is a superset of filter list "
            "(all filter services present in entry). "
            "EXACT: entry scope_services exactly equals filter set "
            "(after normalization: sorted lowercase). "
            "Ignored when scope_services is None or empty."
        ),
    )
    cursor: str | None = Field(
        default=None,
        description=(
            "Opaque pagination cursor from a previous result's next_cursor. "
            "Encodes base64('{created_at_iso}|{decision_id}'). "
            "None = start from beginning (most recent first)."
        ),
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of results per page. Must be between 1 and 1000.",
    )


__all__: list[str] = ["ModelPayloadQueryDecisions"]
