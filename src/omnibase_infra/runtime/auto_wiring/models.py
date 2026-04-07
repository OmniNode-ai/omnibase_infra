# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic models for contract auto-wiring.

Contains lifecycle hooks (OMN-7655) and auto-discovery manifest (OMN-7653).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.config import ModelLifecycleHookConfig

# --- Lifecycle hooks (OMN-7655) ---


class ModelLifecycleHooks(BaseModel):
    """Contract-level lifecycle hooks for auto-wiring.

    Declares optional hooks that the auto-wiring engine invokes during
    node lifecycle transitions. These replace Plugin.initialize() and
    Plugin.shutdown() with contract-declared, auditable callables.

    Phase Ordering:
        1. on_start — called after container wiring, before consumers start
        2. validate_handshake — called after on_start, must pass for wiring
        3. on_shutdown — called during graceful shutdown, before resources close
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    on_start: ModelLifecycleHookConfig | None = Field(
        default=None,
        description="Hook invoked during node startup after container wiring",
    )
    validate_handshake: ModelLifecycleHookConfig | None = Field(
        default=None,
        description="Hook invoked to validate runtime preconditions",
    )
    on_shutdown: ModelLifecycleHookConfig | None = Field(
        default=None,
        description="Hook invoked during graceful node shutdown",
    )

    def has_hooks(self) -> bool:
        """Return True if any lifecycle hook is configured."""
        return any([self.on_start, self.validate_handshake, self.on_shutdown])


# --- Auto-discovery models (OMN-7653 / OMN-7654) ---


class ModelContractVersion(BaseModel):
    """Semantic version extracted from contract YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    major: int = Field(..., description="Major version")
    minor: int = Field(..., description="Minor version")
    patch: int = Field(..., description="Patch version")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class ModelHandlerRef(BaseModel):
    """Reference to a handler class in a contract's handler_routing section."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(..., description="Handler class name")
    module: str = Field(..., description="Fully qualified module path")


class ModelHandlerRoutingEntry(BaseModel):
    """A single handler entry from contract handler_routing.handlers[]."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    handler: ModelHandlerRef = Field(..., description="Handler class reference")
    event_model: ModelHandlerRef | None = Field(
        default=None,
        description="Event model reference (payload_type_match strategy)",
    )
    operation: str | None = Field(
        default=None,
        description="Operation name (operation_match strategy)",
    )


class ModelHandlerRouting(BaseModel):
    """Handler routing declaration from contract YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    routing_strategy: str = Field(
        ..., description="Routing strategy (payload_type_match or operation_match)"
    )
    handlers: tuple[ModelHandlerRoutingEntry, ...] = Field(
        default_factory=tuple,
        description="Handler entries",
    )


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
    entry_point_name: str = Field(..., description="Name of the onex.nodes entry point")
    package_name: str = Field(
        ..., description="Distribution package that registered the entry point"
    )
    package_version: str = Field(
        default="0.0.0", description="Distribution package version"
    )
    event_bus: ModelEventBusWiring | None = Field(
        default=None, description="Event bus wiring if declared"
    )
    handler_routing: ModelHandlerRouting | None = Field(
        default=None, description="Handler routing if declared"
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


__all__ = [
    "ModelAutoWiringManifest",
    "ModelContractVersion",
    "ModelDiscoveredContract",
    "ModelDiscoveryError",
    "ModelEventBusWiring",
    "ModelHandlerRef",
    "ModelHandlerRouting",
    "ModelHandlerRoutingEntry",
    "ModelLifecycleHooks",
]
