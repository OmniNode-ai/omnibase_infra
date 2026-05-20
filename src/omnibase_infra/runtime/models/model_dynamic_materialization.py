# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dynamic contract materialization result model (OMN-11244).

Returned by KafkaContractSource.materialize_cached_contract() when wiring
a cached descriptor into the live dispatch engine post-startup.

Policy enforcement (handler allowlist, runtime profiles, hash verification)
is performed by node_contract_registry in omnimarket. The runtime trusts
validated events on onex.evt.platform.node-registration.v1.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.enums.enum_materialization_rejection import (
    EnumMaterializationRejection,
)
from omnibase_infra.runtime.enums.enum_materialization_status import (
    EnumMaterializationStatus,
)


class ModelDynamicMaterializationResult(BaseModel):
    """Result returned by KafkaContractSource.materialize_cached_contract().

    Structured result with enum status, optional rejection reason, and
    topology data populated on successful materialization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_name: str = Field(..., description="Contract name that was materialized")
    status: EnumMaterializationStatus = Field(
        ..., description="Materialization outcome"
    )
    contract_hash: str = Field(
        default="", description="SHA-256 hash of the contract YAML"
    )
    reason: EnumMaterializationRejection | None = Field(
        default=None, description="Rejection reason (None on success or idempotent)"
    )
    subscribed_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Kafka topics subscribed during materialization",
    )
    registered_handlers: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Handler dispatcher IDs registered during materialization",
    )
    runtime_profile: str = Field(
        default="",
        description="Runtime profile from registration event (populated by omnimarket)",
    )
    materialization_correlation_id: UUID | None = Field(
        default=None,
        description="Correlation UUID for this materialization call",
    )


__all__ = ["ModelDynamicMaterializationResult"]
