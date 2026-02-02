# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic summary model for dashboard display.

Related Tickets:
    - OMN-1845: Contract Registry Persistence
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelTopicSummary(BaseModel):
    """Topic summary for list view.

    Provides a compact view of a topic suitable for list endpoints,
    including direction (publish/subscribe) and contract count.

    Attributes:
        topic_suffix: Topic suffix (without environment prefix)
        direction: Relationship direction ('publish' or 'subscribe')
        contract_count: Number of contracts with this topic relationship
        last_seen_at: Timestamp of last activity on this topic
        is_active: Whether the topic has active contracts
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic_suffix: str = Field(
        ...,
        description="Topic suffix (without environment prefix)",
    )
    direction: str = Field(
        ...,
        description="Relationship direction ('publish' or 'subscribe')",
    )
    contract_count: int = Field(
        ...,
        ge=0,
        description="Number of contracts with this topic relationship",
    )
    last_seen_at: datetime = Field(
        ...,
        description="Timestamp of last activity on this topic",
    )
    is_active: bool = Field(
        ...,
        description="Whether the topic has active contracts",
    )


__all__ = ["ModelTopicSummary"]
