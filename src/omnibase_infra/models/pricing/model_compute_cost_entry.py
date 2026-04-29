# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-GPU compute cost entry from the pricing manifest."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelComputeCostEntry(BaseModel):
    """Hourly compute cost policy for a GPU type.

    Events carry measured usage evidence. This manifest entry carries pricing
    policy applied by projections at read/projection time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    electricity_per_hour: float = Field(
        ...,
        ge=0.0,
        description="Electricity cost in USD per GPU hour.",
    )
    amortization_per_hour: float = Field(
        ...,
        ge=0.0,
        description="Hardware amortization cost in USD per GPU hour.",
    )
    note: str = Field(
        default="",
        max_length=512,
        description="Optional human-readable note about this compute rate.",
    )

    @property
    def total_per_hour(self) -> float:
        """Combined hourly rate in USD."""
        return self.electricity_per_hour + self.amortization_per_hour


__all__: list[str] = ["ModelComputeCostEntry"]
