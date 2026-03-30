# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Savings category breakdown model.

Related Tickets:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_savings_estimation_compute.models.enum_savings_category import (
    EnumSavingsCategory,
)


class ModelSavingsCategory(BaseModel):
    """Savings breakdown by category (pattern type).

    Maps to the JSONB ``categories`` array in the ``savings_estimates``
    table.

    Attributes:
        category: Category name (e.g. 'file', 'architecture', 'tool').
        savings_usd: Total USD saved in this category.
        tokens_saved: Total tokens saved in this category.
        confidence: Confidence score for this category (0.0-1.0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    category: EnumSavingsCategory = Field(..., description="Category name")
    savings_usd: float = Field(..., ge=0.0, description="USD saved")
    tokens_saved: int = Field(..., ge=0, description="Tokens saved")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Category confidence")


__all__: list[str] = ["ModelSavingsCategory"]
