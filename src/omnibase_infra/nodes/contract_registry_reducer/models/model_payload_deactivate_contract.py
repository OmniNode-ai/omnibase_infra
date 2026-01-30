# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Deactivate contract intent payload model.

Related:
    - OMN-1653: Contract Registry Reducer implementation
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadDeactivateContract(BaseModel):
    """Payload for PostgreSQL deactivate contract intents.

    Used when a contract-deregistered event is processed to mark the
    contract as inactive (soft delete).

    Note: contract_id is a derived natural key (node_name:major.minor.patch),
    not a UUID. This is intentional per the contract registry design.

    Attributes:
        intent_type: Routing discriminator. Always "postgres.deactivate_contract".
        correlation_id: Correlation ID for distributed tracing.
        contract_id: Contract to deactivate.
        node_name: Contract node name.
        reason: Deregistration reason (shutdown, upgrade, manual).
        deactivated_at: Deactivation timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["postgres.deactivate_contract"] = Field(
        default="postgres.deactivate_contract",
        description="Routing discriminator for intent dispatch.",
    )

    correlation_id: UUID = Field(
        ..., description="Correlation ID for distributed tracing."
    )

    # ONEX_EXCLUDE: pattern_validator - contract_id is a derived natural key (name:version), not UUID
    contract_id: str = Field(..., description="Contract to deactivate.")

    # ONEX_EXCLUDE: pattern_validator - node_name is the contract name, not an entity reference
    node_name: str = Field(..., description="Contract node name.")

    reason: str = Field(..., description="Deregistration reason.")

    deactivated_at: datetime = Field(..., description="Deactivation timestamp.")


__all__ = ["ModelPayloadDeactivateContract"]
