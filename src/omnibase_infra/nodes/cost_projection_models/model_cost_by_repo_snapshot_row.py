# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost-by-repository projection snapshot row model."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ModelCostByRepoSnapshotRow(BaseModel):
    """One repository bucket in a cost-by-repo snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo_name: str | None = Field(default=None)
    cost_usd: Decimal = Field(ge=Decimal("0"))
    total_tokens: int = Field(ge=0)
    call_count: int = Field(ge=0)


__all__ = ["ModelCostByRepoSnapshotRow"]
