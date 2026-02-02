# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract view model for dashboard display.

Related Tickets:
    - OMN-1845: Contract Registry Persistence
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelContractView(BaseModel):
    """Contract detail for API responses.

    Represents a registered contract from the contract registry,
    flattened for dashboard consumption.

    Attributes:
        contract_id: Unique identifier in format node_name:major.minor.patch
        node_name: Name of the node this contract belongs to
        version: Semantic version string in format major.minor.patch
        contract_hash: SHA-256 hash of contract content for integrity verification
        is_active: Whether the contract is currently active
        registered_at: Timestamp of initial registration
        last_seen_at: Timestamp of last activity (heartbeat or event)
        deregistered_at: Timestamp when contract was deregistered (None if active)
        topics_published: List of topic suffixes this contract publishes to
        topics_subscribed: List of topic suffixes this contract subscribes to
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # ONEX_EXCLUDE: pattern_validator - contract_id is a derived natural key (name:version), not UUID
    contract_id: str = Field(
        ...,
        description="Unique identifier in format node_name:major.minor.patch",
    )
    # ONEX_EXCLUDE: pattern_validator - node_name is the contract name, not an entity reference
    node_name: str = Field(
        ...,
        description="Name of the node this contract belongs to",
    )
    version: str = Field(
        ...,
        description="Semantic version string in format major.minor.patch",
    )
    contract_hash: str = Field(
        ...,
        description="SHA-256 hash of contract content for integrity verification",
    )
    is_active: bool = Field(
        ...,
        description="Whether the contract is currently active",
    )
    registered_at: datetime = Field(
        ...,
        description="Timestamp of initial registration",
    )
    last_seen_at: datetime = Field(
        ...,
        description="Timestamp of last activity (heartbeat or event)",
    )
    deregistered_at: datetime | None = Field(
        default=None,
        description="Timestamp when contract was deregistered (None if active)",
    )
    topics_published: list[str] = Field(
        default_factory=list,
        description="List of topic suffixes this contract publishes to",
    )
    topics_subscribed: list[str] = Field(
        default_factory=list,
        description="List of topic suffixes this contract subscribes to",
    )


__all__ = ["ModelContractView"]
