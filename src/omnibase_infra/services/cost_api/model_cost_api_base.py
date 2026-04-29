# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared base model for LLM cost API responses."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_serializer


class ModelCostApiBase(BaseModel):
    """Base model with strict, immutable API contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    @field_serializer(
        "total_cost_usd",
        "total_savings_usd",
        "local_cost_usd",
        "cloud_cost_usd",
        "savings_usd",
        "estimated_coverage_pct",
        "average_tokens_per_call",
        check_fields=False,
    )
    def serialize_decimal(self, value: Decimal | None) -> str | None:
        """Serialize Decimal values as strings to avoid JSON float drift."""
        if value is None:
            return None
        return str(value)
