# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic view model for dashboard display.

Related Tickets:
    - OMN-1845: Contract Registry Persistence
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.registry_api.models.model_contract_ref import (
    ModelContractRef,
)


class ModelTopicView(BaseModel):
    """Topic detail for API responses.

    Represents a topic with its publishers and subscribers,
    providing full detail for topic inspection endpoints.

    Attributes:
        topic_suffix: Topic suffix (without environment prefix)
        publishers: List of contracts that publish to this topic
        subscribers: List of contracts that subscribe to this topic
        first_seen_at: Timestamp when topic was first observed
        last_seen_at: Timestamp of last activity on this topic
        is_active: Whether the topic has active publishers or subscribers
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic_suffix: str = Field(
        ...,
        description="Topic suffix (without environment prefix)",
    )
    publishers: list[ModelContractRef] = Field(
        default_factory=list,
        description="List of contracts that publish to this topic",
    )
    subscribers: list[ModelContractRef] = Field(
        default_factory=list,
        description="List of contracts that subscribe to this topic",
    )
    first_seen_at: datetime = Field(
        ...,
        description="Timestamp when topic was first observed",
    )
    last_seen_at: datetime = Field(
        ...,
        description="Timestamp of last activity on this topic",
    )
    is_active: bool = Field(
        ...,
        description="Whether the topic has active publishers or subscribers",
    )


__all__ = ["ModelTopicView"]
