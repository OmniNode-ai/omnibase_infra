# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic Projection Model.

Provides the Pydantic model for topic projections stored in PostgreSQL.
Used by the Registry API to query topic suffixes referenced by contracts
for routing discovery.

Topics use 5-segment naming (e.g., onex.evt.platform.contract-registered.v1)
and store SUFFIXES only - environment prefix is applied at runtime.

Related Tickets:
    - OMN-1845: Create ProjectionReaderContract for contract/topic queries
    - OMN-1653: Contract registry state materialization
    - OMN-1709: Topic orphan handling documentation
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Valid direction values for topic routing
TopicDirection = Literal["publish", "subscribe"]


class ModelTopicProjection(BaseModel):
    """Topic projection for Registry API queries.

    Represents a topic suffix referenced by contracts for routing discovery.
    This model maps to the topics table created by migration 005.

    Primary Key:
        (topic_suffix, direction) - composite key allowing same topic with
        both publish and subscribe directions

    Topic Orphan Handling (OMN-1709):
        When all contracts referencing a topic are deregistered, the topic
        record remains with an empty contract_ids list. This is intentional:
        - Preserves topic routing history for auditing and debugging
        - Allows topic reactivation if a new contract references the same topic
        - Avoids complex cascading deletes during high-volume deregistration

    Attributes:
        topic_suffix: Topic suffix without environment prefix
        direction: Whether contracts publish to or subscribe from this topic
        contract_ids: List of contract_id strings that reference this topic
        first_seen_at: Timestamp when topic was first seen in any contract
        last_seen_at: Timestamp when topic was last seen in any contract
        is_active: Whether topic is currently referenced by any active contract
        created_at: Timestamp when row was created
        updated_at: Timestamp when row was last updated

    Example:
        >>> from datetime import datetime, UTC
        >>> now = datetime.now(UTC)
        >>> projection = ModelTopicProjection(
        ...     topic_suffix="onex.evt.platform.contract-registered.v1",
        ...     direction="publish",
        ...     contract_ids=["node-registry-effect:1.0.0"],
        ...     first_seen_at=now,
        ...     last_seen_at=now,
        ...     is_active=True,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Identity (composite primary key)
    topic_suffix: str = Field(
        ...,
        min_length=1,
        description=(
            "Topic suffix without environment prefix, "
            "e.g., onex.evt.platform.contract-registered.v1"
        ),
    )
    direction: TopicDirection = Field(
        ...,
        description="Whether contracts publish to or subscribe from this topic",
    )

    # Contract references (JSONB array in database)
    contract_ids: list[str] = Field(
        default_factory=list,
        description="List of contract_id strings that reference this topic",
    )

    # Lifecycle
    first_seen_at: datetime = Field(
        ...,
        description="Timestamp when topic was first seen in any contract",
    )
    last_seen_at: datetime = Field(
        ...,
        description="Timestamp when topic was last seen in any contract",
    )
    is_active: bool = Field(
        default=True,
        description="Whether topic is currently referenced by any active contract",
    )

    # Audit timestamps
    created_at: datetime | None = Field(
        default=None,
        description="Timestamp when row was created",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Timestamp when row was last updated",
    )

    @property
    def contract_count(self) -> int:
        """Return number of contracts referencing this topic.

        Returns:
            Count of contract IDs

        Example:
            >>> proj = ModelTopicProjection(contract_ids=["a:1.0.0", "b:1.0.0"], ...)
            >>> proj.contract_count
            2
        """
        return len(self.contract_ids)

    @property
    def is_orphaned(self) -> bool:
        """Check if topic is orphaned (no contracts reference it).

        Returns:
            True if contract_ids is empty and topic is inactive

        Example:
            >>> proj = ModelTopicProjection(contract_ids=[], is_active=False, ...)
            >>> proj.is_orphaned
            True
        """
        return len(self.contract_ids) == 0 and not self.is_active


__all__: list[str] = ["ModelTopicProjection", "TopicDirection"]
