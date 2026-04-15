# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wiring manifest produced by contract discovery (OMN-7653)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovery_error import (
    ModelDiscoveryError,
)


class ModelAutoWiringManifest(BaseModel):
    """Complete manifest produced by contract auto-discovery.

    Contains all successfully discovered contracts and any errors
    encountered during scanning. Pure data — no side effects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contracts: tuple[ModelDiscoveredContract, ...] = Field(
        default_factory=tuple,
        description="Successfully discovered contracts",
    )
    errors: tuple[ModelDiscoveryError, ...] = Field(
        default_factory=tuple,
        description="Errors encountered during discovery",
    )

    @property
    def total_discovered(self) -> int:
        return len(self.contracts)

    @property
    def total_errors(self) -> int:
        return len(self.errors)

    def get_by_node_type(self, node_type: str) -> tuple[ModelDiscoveredContract, ...]:
        """Filter discovered contracts by node type."""
        return tuple(c for c in self.contracts if c.node_type == node_type)

    def get_all_subscribe_topics(self) -> frozenset[str]:
        """Collect all subscribe topics across discovered contracts."""
        topics: set[str] = set()
        for c in self.contracts:
            if c.event_bus:
                topics.update(c.event_bus.subscribe_topics)
        return frozenset(topics)

    def all_subscribe_topics(self) -> frozenset[str]:
        """Alias satisfying ProtocolAutoWiringManifestLike (OMN-8854)."""
        return self.get_all_subscribe_topics()

    def get_all_publish_topics(self) -> frozenset[str]:
        """Collect all publish topics across discovered contracts."""
        topics: set[str] = set()
        for c in self.contracts:
            if c.event_bus:
                topics.update(c.event_bus.publish_topics)
        return frozenset(topics)
