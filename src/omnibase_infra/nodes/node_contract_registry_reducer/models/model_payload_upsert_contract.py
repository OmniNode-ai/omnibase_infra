# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Upsert contract intent payload model.

Related:
    - OMN-1653: Contract Registry Reducer implementation
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelPayloadUpsertContract(BaseModel):
    """Payload for PostgreSQL upsert contract intents.

    Used when a contract-registered event is processed to insert or update
    the contract record in the contracts table.

    Note: contract_id is a derived natural key (node_name:major.minor.patch),
    not a UUID. This is intentional per the contract registry design.

    Serialization Note:
        The contract_yaml field accepts both dict (parsed) and str (raw YAML).
        The Effect layer (PostgresAdapter) is responsible for serializing dict
        to YAML string before INSERT, as the PostgreSQL column is TEXT type.

    Attributes:
        intent_type: Routing discriminator. Always "postgres.upsert_contract".
        correlation_id: Correlation ID for distributed tracing.
        contract_id: Derived contract identity (node_name:major.minor.patch).
        node_name: Contract node name from event.
        version_major: Semantic version major component.
        version_minor: Semantic version minor component.
        version_patch: Semantic version patch component.
        contract_hash: Hash of the contract YAML for change detection.
        contract_yaml: Full contract YAML for storage and replay. Accepts dict
            (parsed) or str (raw). Effect layer serializes dict to YAML string
            before INSERT.
        source_node_id: UUID of the node that emitted the event (optional).
        is_active: Whether the contract is currently active.
        registered_at: Timestamp when contract was registered.
        last_seen_at: Last heartbeat/registration timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["postgres.upsert_contract"] = Field(
        default="postgres.upsert_contract",
        description="Routing discriminator for intent dispatch.",
    )

    correlation_id: UUID = Field(
        ..., description="Correlation ID for distributed tracing."
    )

    # ONEX_EXCLUDE: pattern_validator - contract_id is a derived natural key (name:version), not UUID
    contract_id: str = Field(
        ..., description="Contract identity: node_name:major.minor.patch"
    )

    # ONEX_EXCLUDE: pattern_validator - node_name is the contract name, not an entity reference
    node_name: str = Field(..., description="Contract node name.")

    version_major: int = Field(..., ge=0, description="Semantic version major.")
    version_minor: int = Field(..., ge=0, description="Semantic version minor.")
    version_patch: int = Field(..., ge=0, description="Semantic version patch.")

    contract_hash: str = Field(..., description="Hash of contract YAML.")

    contract_yaml: JsonType = Field(
        ..., description="Full contract YAML (dict or raw string)."
    )

    source_node_id: str | None = Field(
        default=None, description="Source node UUID (optional)."
    )

    is_active: bool = Field(default=True, description="Whether contract is active.")

    registered_at: datetime = Field(..., description="Registration timestamp.")

    last_seen_at: datetime = Field(..., description="Last heartbeat timestamp.")


__all__ = ["ModelPayloadUpsertContract"]
