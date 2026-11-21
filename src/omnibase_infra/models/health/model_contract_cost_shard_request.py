# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to rank the stored contract cost readings for a shard split (OMN-26250)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.model_contract_cost_reading import (
    ModelContractCostReading,
)


class ModelContractCostShardRequest(BaseModel):
    """The stored readings of a period and the contracts the runtime serves.

    ``served_contracts`` is the manifest the readings are joined to: a served
    contract with no reading is reported, not dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    readings: tuple[ModelContractCostReading, ...] = Field(
        ..., min_length=1, description="Stored per-window contract cost readings."
    )
    served_contracts: tuple[str, ...] = Field(
        ..., description="Names of the contracts the runtime attaches."
    )


__all__: list[str] = ["ModelContractCostShardRequest"]
