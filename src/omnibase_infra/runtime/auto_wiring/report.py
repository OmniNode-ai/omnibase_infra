# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wiring report models for OMN-7654.

Captures per-contract wiring outcomes: success, skip, or failure with reason.

Per-handler resolver outcomes and skip entries (added for OMN-9201) are
modeled on :class:`ModelContractWiringResult` so the wiring report can cite
which handler the resolver skipped and why, without expanding the error
surface. See
``docs/plans/2026-04-18-handler-resolver-architecture.md`` Task 5 §Report
schema ownership.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)


class EnumWiringOutcome(str, Enum):
    """Outcome of wiring a single discovered contract."""

    WIRED = "wired"
    SKIPPED = "skipped"
    FAILED = "failed"


class ModelWiringOutcome(BaseModel):
    """Per-handler resolver outcome row within a contract-level wiring result.

    Populated from :class:`omnibase_core.models.resolver.ModelHandlerResolution`
    returned by ``ServiceHandlerResolver.resolve(...)`` at wiring time.

    See ``docs/plans/2026-04-18-handler-resolver-architecture.md`` §Known Types
    Inventory for why this is not reused from :class:`ModelContractWiringResult`
    (different granularity) or :class:`ModelHandlerResolution` (that is the
    core-layer resolver return; this is the infra-layer report projection
    used for observability / determinism asserts).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    handler_name: str = Field(
        ..., description="Handler class name (from contract.handler_routing)"
    )
    resolution_outcome: EnumHandlerResolutionOutcome = Field(
        ..., description="Which resolver precedence path produced this handler"
    )
    skipped_reason: str = Field(
        default="",
        description="Skip reason (empty string on successful resolution)",
    )


class ModelSkippedEntry(BaseModel):
    """Skip-only record surfaced at the contract level.

    See ``docs/plans/2026-04-18-handler-resolver-architecture.md`` §Known Types
    Inventory: this is narrower than :class:`ModelWiringOutcome`
    because the outcomes list records every resolved handler, while this list
    is the filtered skip subset surfaced at the contract level.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    handler_name: str = Field(..., description="Handler class name skipped")
    reason: str = Field(..., description="Skip reason (human-readable)")


class ModelContractWiringResult(BaseModel):
    """Wiring result for a single discovered contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_name: str = Field(..., description="Node name from contract")
    package_name: str = Field(..., description="Package that owns this contract")
    outcome: EnumWiringOutcome = Field(..., description="Wiring outcome")
    reason: str = Field(default="", description="Reason for skip or failure")
    dispatchers_registered: tuple[str, ...] = Field(
        default_factory=tuple, description="Dispatcher IDs registered"
    )
    routes_registered: tuple[str, ...] = Field(
        default_factory=tuple, description="Route IDs registered"
    )
    topics_subscribed: tuple[str, ...] = Field(
        default_factory=tuple, description="Kafka topics subscribed"
    )
    wirings: tuple[ModelWiringOutcome, ...] = Field(
        default_factory=tuple,
        description=(
            "Per-handler resolver outcomes recorded by "
            "ServiceHandlerResolver (OMN-9201). Order mirrors the contract's "
            "handler_routing entries."
        ),
    )
    skipped_handlers: tuple[ModelSkippedEntry, ...] = Field(
        default_factory=tuple,
        description=(
            "Handlers skipped by the resolver's "
            "RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP outcome (OMN-9201). "
            "Not errors."
        ),
    )


class ModelDuplicateTopicOwnership(BaseModel):
    """A duplicate topic subscription detected during wiring."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = Field(..., description="The topic with duplicate ownership")
    owners: tuple[str, ...] = Field(
        ..., description="Contract names that subscribe to this topic"
    )
    level: str = Field(
        ...,
        description="Duplicate level: package, handler, or intra-package",
    )


class ModelAutoWiringReport(BaseModel):
    """Complete report produced by the auto-wiring engine.

    Contains per-contract outcomes, duplicate topic warnings, and aggregate
    statistics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    results: tuple[ModelContractWiringResult, ...] = Field(
        default_factory=tuple, description="Per-contract wiring results"
    )
    duplicates: tuple[ModelDuplicateTopicOwnership, ...] = Field(
        default_factory=tuple, description="Duplicate topic ownership warnings"
    )

    @property
    def total_wired(self) -> int:
        return sum(1 for r in self.results if r.outcome == EnumWiringOutcome.WIRED)

    @property
    def total_skipped(self) -> int:
        return sum(1 for r in self.results if r.outcome == EnumWiringOutcome.SKIPPED)

    @property
    def total_failed(self) -> int:
        return sum(1 for r in self.results if r.outcome == EnumWiringOutcome.FAILED)

    def __bool__(self) -> bool:
        """True when all contracts wired or skipped (no failures).

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            there are zero failures. Differs from typical Pydantic behavior.
        """
        return self.total_failed == 0
