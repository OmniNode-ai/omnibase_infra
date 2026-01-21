# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pagination info model for list endpoints.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPaginationInfo(BaseModel):
    """Pagination information for list endpoints.

    Attributes:
        total: Total number of items matching the query
        limit: Maximum items per page
        offset: Current offset (0-based)
        has_more: Whether more items exist beyond current page
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(
        ...,
        ge=0,
        description="Total number of items",
    )
    limit: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Maximum items per page",
    )
    offset: int = Field(
        ...,
        ge=0,
        description="Current offset (0-based)",
    )
    has_more: bool = Field(
        ...,
        description="Whether more items exist",
    )


__all__ = ["ModelPaginationInfo"]
