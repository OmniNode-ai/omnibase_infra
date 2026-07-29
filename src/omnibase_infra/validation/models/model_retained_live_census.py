# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-catalog completeness counters from an ownership manifest."""

from pydantic import BaseModel, ConfigDict, Field


class ModelRetainedLiveCensus(BaseModel):
    """Completeness counters projected by migration inventory manifests."""

    model_config = ConfigDict(frozen=True, extra="allow")

    observed_base_tables: int | None = Field(default=None, ge=0)
    observed_views_and_materialized_views: int | None = Field(default=None, ge=0)
    parity_status: str | None = None
    reason: str | None = None


__all__ = ["ModelRetainedLiveCensus"]
