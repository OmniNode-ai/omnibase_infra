# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Cleanup topic references intent payload model.

Related:
    - OMN-1653: Contract Registry Reducer implementation
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadCleanupTopicReferences(BaseModel):
    """Payload for PostgreSQL cleanup topic references intents.

    Used when a contract-deregistered event is processed to remove
    the contract_id from all topics.contract_ids JSONB arrays.

    Note: contract_id is a derived natural key (node_name:major.minor.patch),
    not a UUID. This is intentional per the contract registry design.

    Attributes:
        intent_type: Routing discriminator. Always "postgres.cleanup_topic_references".
        correlation_id: Correlation ID for distributed tracing.
        contract_id: Contract ID to remove from all topic references.
        node_name: Contract node name (for logging/debugging).
        cleaned_at: Cleanup timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["postgres.cleanup_topic_references"] = Field(
        default="postgres.cleanup_topic_references",
        description="Routing discriminator for intent dispatch.",
    )

    correlation_id: UUID = Field(
        ..., description="Correlation ID for distributed tracing."
    )

    # ONEX_EXCLUDE: pattern_validator - contract_id is a derived natural key (name:version), not UUID
    contract_id: str = Field(..., description="Contract ID to remove from topics.")

    # ONEX_EXCLUDE: pattern_validator - node_name is the contract name, not an entity reference
    node_name: str = Field(..., description="Contract node name.")

    cleaned_at: datetime = Field(..., description="Cleanup timestamp.")


__all__ = ["ModelPayloadCleanupTopicReferences"]
