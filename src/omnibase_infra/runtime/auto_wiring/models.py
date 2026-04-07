# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic models for contract auto-discovery, auto-wiring, and lifecycle hooks.

Includes:
- Discovery models: contract manifest, handler routing refs (OMN-7653, OMN-7654)
- Lifecycle hook models: hook config, results, handshake, quarantine (OMN-7655, OMN-7657)
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Discovery models (OMN-7653, OMN-7654)
# ---------------------------------------------------------------------------


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
    lifecycle_hooks: ModelLifecycleHooks | None = Field(
        default=None, description="Lifecycle hooks if declared"
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


# ---------------------------------------------------------------------------
# Lifecycle hook models (OMN-7655, OMN-7657)
# ---------------------------------------------------------------------------


class ModelLifecycleHookConfig(BaseModel):
    """Configuration for a single lifecycle hook callable.

    Represents a dotted-path reference to an async callable that will be
    invoked during the corresponding lifecycle phase. The callable receives
    a ModelAutoWiringContext and returns a ModelLifecycleHookResult.

    Attributes:
        callable_ref: Dotted import path to the async hook callable.
            Example: ``mypackage.hooks.on_start_handler``
        timeout_seconds: Maximum execution time before the hook is cancelled.
        required: If True, hook failure aborts the lifecycle phase.
            If False, failure is logged but does not block.
        idempotent: Assertion that this hook is safe to retry. Must be True.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    callable_ref: str = Field(
        ...,
        min_length=1,
        description="Dotted import path to the async hook callable",
    )
    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,
        description="Maximum execution time in seconds",
    )
    required: bool = Field(
        default=True,
        description="Whether hook failure aborts the lifecycle phase",
    )
    idempotent: bool = Field(
        default=True,
        description="Assertion that this hook is safe to retry (must be True)",
    )

    @model_validator(mode="after")
    def validate_idempotent(self) -> Self:
        """Enforce that all lifecycle hooks declare themselves as idempotent."""
        if not self.idempotent:
            msg = (
                f"Lifecycle hook '{self.callable_ref}' must declare "
                "idempotent=True. Non-idempotent hooks are not permitted."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_callable_ref_format(self) -> Self:
        """Validate that callable_ref looks like a dotted Python import path."""
        parts = self.callable_ref.split(".")
        if len(parts) < 2:
            msg = (
                f"callable_ref '{self.callable_ref}' must be a dotted path "
                "with at least two segments (e.g., 'package.function')"
            )
            raise ValueError(msg)
        for part in parts:
            if not part.isidentifier():
                msg = (
                    f"callable_ref segment '{part}' in '{self.callable_ref}' "
                    "is not a valid Python identifier"
                )
                raise ValueError(msg)
        return self


class HandshakeFailureReason(str, Enum):
    """Structured reasons for handshake failure."""

    TIMEOUT = "timeout"
    RESOLUTION_FAILED = "resolution_failed"
    DB_OWNERSHIP = "db_ownership"
    SCHEMA_FINGERPRINT = "schema_fingerprint"
    TCP_PROBE_FAILED = "tcp_probe_failed"
    HOOK_EXCEPTION = "hook_exception"
    HOOK_RETURNED_FAILURE = "hook_returned_failure"


class ModelHandshakeConfig(BaseModel):
    """Configuration for handshake retry and timeout behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum retry attempts after initial failure",
    )
    retry_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=60.0,
        description="Delay between retry attempts in seconds",
    )
    total_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="Overall deadline for all handshake attempts",
    )


class ModelQuarantineRecord(BaseModel):
    """Record of a quarantined contract that failed handshake validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    handler_id: str = Field(
        ...,
        min_length=1,
        description="Handler ID of the quarantined contract",
    )
    node_kind: str = Field(
        ...,
        min_length=1,
        description="Node kind of the quarantined contract",
    )
    failure_reason: HandshakeFailureReason = Field(
        ...,
        description="Classified reason for quarantine",
    )
    error_message: str = Field(
        default="",
        description="Diagnostic message from the last failed attempt",
    )
    attempts: int = Field(
        ...,
        ge=1,
        description="Total number of handshake attempts made",
    )
    quarantined_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="UTC timestamp when the contract was quarantined",
    )


class ModelLifecycleHooks(BaseModel):
    """Contract-level lifecycle hooks for auto-wiring.

    Phase Ordering:
        1. on_start -- called after container wiring, before consumers start
        2. validate_handshake -- called after on_start, must pass for wiring
        3. on_shutdown -- called during graceful shutdown, before resources close
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
    handshake_config: ModelHandshakeConfig = Field(
        default_factory=ModelHandshakeConfig,
        description="Retry and timeout configuration for handshake validation",
    )
    on_shutdown: ModelLifecycleHookConfig | None = Field(
        default=None,
        description="Hook invoked during graceful node shutdown",
    )

    def has_hooks(self) -> bool:
        """Return True if any lifecycle hook is configured."""
        return any([self.on_start, self.validate_handshake, self.on_shutdown])


class ModelLifecycleHookResult(BaseModel):
    """Structured result from a lifecycle hook execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hook_name: str = Field(
        ...,
        min_length=1,
        description="Lifecycle phase name (on_start, validate_handshake, on_shutdown)",
    )
    success: bool = Field(
        ...,
        description="Whether the hook completed successfully",
    )
    error_message: str = Field(
        default="",
        description="Diagnostic message if the hook failed",
    )
    background_workers: list[str] = Field(
        default_factory=list,
        description="Names of background tasks started by this hook",
    )

    @classmethod
    def succeeded(
        cls,
        hook_name: str,
        background_workers: list[str] | None = None,
    ) -> ModelLifecycleHookResult:
        """Create a successful hook result."""
        return cls(
            hook_name=hook_name,
            success=True,
            background_workers=background_workers or [],
        )

    @classmethod
    def failed(cls, hook_name: str, error_message: str) -> ModelLifecycleHookResult:
        """Create a failed hook result."""
        return cls(
            hook_name=hook_name,
            success=False,
            error_message=error_message,
        )

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success
