# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Record Model.

This module provides the ModelRegistrationRecord class representing a node
registration record in persistent storage.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from pydantic import BaseModel, ConfigDict, Field


class ModelRegistrationRecord(BaseModel):
    """Node registration record for persistent storage.

    Represents a node registration record with its metadata for storage
    operations. Immutable once created.

    Attributes:
        node_id: Unique identifier for the node.
        node_type: Type of ONEX node (EnumNodeKind).
        node_version: Semantic version of the node.
        endpoints: Dict of endpoint type to URL.
        metadata: Additional key-value metadata.
        created_at: Timestamp when the record was created.
        updated_at: Timestamp when the record was last updated.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: UUID = Field(
        ...,
        description="Unique identifier for the node",
    )
    node_type: EnumNodeKind = Field(
        ...,
        description="Type of ONEX node",
    )
    node_version: str = Field(
        ...,
        description="Semantic version of the node",
        min_length=1,
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Dict of endpoint type to URL",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Timestamp when the record was created",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Timestamp when the record was last updated",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )


__all__ = ["ModelRegistrationRecord"]
