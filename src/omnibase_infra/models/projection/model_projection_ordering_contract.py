# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection ordering contracts for materialized table readers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.projection.enum_projection_ordering_direction import (
    EnumProjectionOrderingDirection,
)


class ModelProjectionOrderingContract(BaseModel):
    """Declares authoritative ordering fields for a projected table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_table: str = Field(..., min_length=1)
    primary_order_field: str = Field(..., min_length=1)
    tie_breaker_field: str = Field(..., min_length=1)
    direction: EnumProjectionOrderingDirection
    non_authoritative_fields: tuple[str, ...] = Field(default_factory=tuple)


DISPATCH_EVAL_RESULTS_ORDERING_CONTRACT = ModelProjectionOrderingContract(
    projection_table="dispatch_eval_results",
    primary_order_field="evaluated_at",
    tie_breaker_field="dispatch_id",
    direction=EnumProjectionOrderingDirection.DESCENDING,
    non_authoritative_fields=("created_at",),
)

PROJECTION_ORDERING_CONTRACTS: tuple[ModelProjectionOrderingContract, ...] = (
    DISPATCH_EVAL_RESULTS_ORDERING_CONTRACT,
)


def get_projection_ordering_contract(
    projection_table: str,
) -> ModelProjectionOrderingContract | None:
    """Return the registered ordering contract for a table, if present."""
    for contract in PROJECTION_ORDERING_CONTRACTS:
        if contract.projection_table == projection_table:
            return contract
    return None


__all__ = [
    "DISPATCH_EVAL_RESULTS_ORDERING_CONTRACT",
    "EnumProjectionOrderingDirection",
    "ModelProjectionOrderingContract",
    "PROJECTION_ORDERING_CONTRACTS",
    "get_projection_ordering_contract",
]
