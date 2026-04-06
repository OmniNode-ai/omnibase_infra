# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic models for contract auto-discovery and auto-wiring manifest."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelContractVersion(BaseModel):
    """Semantic version extracted from contract YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    major: int = Field(..., description="Major version")
    minor: int = Field(..., description="Minor version")
    patch: int = Field(..., description="Patch version")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class ModelEventBusWiring(BaseModel):
    """Event bus topic declarations extracted from a contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    subscribe_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topics this node subscribes to",
    )
    publish_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topics this node publishes to",
    )


class ModelDiscoveredContract(BaseModel):
    """A single contract discovered from an onex.nodes entry point.

    Captures the subset of contract YAML fields needed for auto-wiring
    without importing any handler or node classes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(..., description="Node name from contract")
    node_type: str = Field(..., description="Node type (e.g. EFFECT_GENERIC)")
    description: str = Field(default="", description="Node description")
    contract_version: ModelContractVersion = Field(
        ..., description="Contract semantic version"
    )
    node_version: str = Field(default="1.0.0", description="Node version string")
    contract_path: Path = Field(..., description="Filesystem path to contract.yaml")
    entry_point_name: str = Field(
        ..., description="Name of the onex.nodes entry point"
    )
    package_name: str = Field(
        ..., description="Distribution package that registered the entry point"
    )
    package_version: str = Field(
        default="0.0.0", description="Distribution package version"
    )
    event_bus: ModelEventBusWiring | None = Field(
        default=None, description="Event bus wiring if declared"
    )


class ModelDiscoveryError(BaseModel):
    """An error encountered during contract discovery."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    entry_point_name: str = Field(..., description="Entry point that failed")
    package_name: str = Field(default="unknown", description="Package name")
    error: str = Field(..., description="Error message")


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

    def get_all_publish_topics(self) -> frozenset[str]:
        """Collect all publish topics across discovered contracts."""
        topics: set[str] = set()
        for c in self.contracts:
            if c.event_bus:
                topics.update(c.event_bus.publish_topics)
        return frozenset(topics)
