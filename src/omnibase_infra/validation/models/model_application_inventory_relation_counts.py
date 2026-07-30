# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed aggregate counts from the OMN-15423 inventory projection."""

from pydantic import BaseModel, ConfigDict, Field


class ModelApplicationInventoryRelationCounts(BaseModel):
    """Counts retained for completeness checks across every projected kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: int = Field(..., ge=0)
    view: int = Field(..., ge=0)
    materialized_view: int = Field(default=0, ge=0)
    function: int = Field(..., ge=0)
    procedure: int | None = Field(default=None, ge=0)
    sequence: int = Field(..., ge=0)
    extension: int = Field(..., ge=0)
    type: int | None = Field(default=None, ge=0)


__all__ = ["ModelApplicationInventoryRelationCounts"]
