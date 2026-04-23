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
from omnibase_infra.runtime.auto_wiring.enum_quarantine_reason import (
    EnumQuarantineReason,
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


class ModelQuarantinedWiring(BaseModel):
    """Deterministically quarantined handler surfaced at the contract level.

    Produced by the auto-wiring engine when handler construction raises a
    known containment-worthy error (e.g. ``asyncio.run()`` called from a
    running event loop — see :class:`EnumQuarantineReason` for the closed
    set of categories).

    Quarantined handlers are reported visibly (not silently skipped) so
    follow-up migration tickets are obvious (OMN-9457). They do NOT poison
    runtime boot: the containing contract still reports ``WIRED`` if its
    remaining handlers resolve, and ``SKIPPED`` only if every handler is
    quarantined.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_name: str = Field(..., description="Contract name owning the handler")
    package_name: str = Field(..., description="Package name owning the handler")
    handler_module: str = Field(..., description="Fully-qualified handler module path")
    handler_name: str = Field(..., description="Handler class name")
    reason: EnumQuarantineReason = Field(
        ..., description="Structured quarantine reason (closed set)"
    )
    detail: str = Field(
        default="",
        description=(
            "Sanitized one-line error detail. Safe for logging and reporting — "
            "URLs / DSNs are redacted. Empty when no additional detail is "
            "available."
        ),
    )


class ModelContractWiringResult(BaseModel):
    """Wiring result for a single discovered contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_name: str = Field(..., description="Node name from contract")
    package_name: str = Field(..., description="Package that owns this contract")
    outcome: EnumWiringOutcome = Field(..., description="Wiring outcome")
    reason: str = Field(default="", description="Reason for skip or failure")
    runtime_profile: str = Field(
        default="default",
        description="Runtime profile under which this contract was evaluated.",
    )
    profile_optional: bool = Field(
        default=False,
        description="True when the contract is optional for the active runtime profile.",
    )
    profile_skip_reason: str = Field(
        default="",
        description=(
            "Structured skip bucket for startup-state reporting "
            "(e.g. 'ineligible', 'optional_unresolved')."
        ),
    )
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
    quarantined_handlers: tuple[ModelQuarantinedWiring, ...] = Field(
        default_factory=tuple,
        description=(
            "Handlers deterministically contained during construction "
            "(OMN-9457). Most common cause: async-incompatible handlers that "
            "call asyncio.run() inside runtime-managed async boot. Quarantined "
            "handlers do not poison startup but require follow-up migration. "
            "A contract whose remaining handlers resolve cleanly still "
            "reports WIRED; a contract whose every handler is quarantined "
            "reports SKIPPED with reason='all handlers quarantined'."
        ),
    )
    structural_invalid_handlers: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Handlers rejected before dispatch because they lack the required "
            "runtime callable contract."
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
    runtime_profile: str = Field(
        default="default",
        description="Runtime profile under which the report was produced.",
    )
    duplicates: tuple[ModelDuplicateTopicOwnership, ...] = Field(
        default_factory=tuple, description="Duplicate topic ownership warnings"
    )
    quarantined_handlers: tuple[ModelQuarantinedWiring, ...] = Field(
        default_factory=tuple,
        description=(
            "Flat list of every quarantined handler across all contracts "
            "(OMN-9457). Mirrors the per-contract "
            "ModelContractWiringResult.quarantined_handlers collections so "
            "follow-up migration tickets can enumerate the full set without "
            "walking every result."
        ),
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

    @property
    def total_quarantined(self) -> int:
        """Count of handlers quarantined during wiring (OMN-9457)."""
        return len(self.quarantined_handlers)

    @property
    def optional_skipped_contracts(self) -> int:
        return sum(
            1 for r in self.results if r.profile_skip_reason == "optional_unresolved"
        )

    @property
    def ineligible_skipped_contracts(self) -> int:
        return sum(1 for r in self.results if r.profile_skip_reason == "ineligible")

    @property
    def structural_invalid_handler_count(self) -> int:
        return sum(len(r.structural_invalid_handlers) for r in self.results)

    @property
    def mandatory_unresolved_contracts(self) -> int:
        return sum(
            1
            for r in self.results
            if r.outcome == EnumWiringOutcome.FAILED and not r.profile_optional
        )

    @property
    def startup_state(self) -> str:
        if self.total_failed > 0 or self.mandatory_unresolved_contracts > 0:
            return "failed"
        if (
            self.optional_skipped_contracts > 0
            or self.total_quarantined > 0
            or self.structural_invalid_handler_count > 0
        ):
            return "degraded"
        return "healthy"

    def startup_summary(self) -> dict[str, object]:
        return {
            "runtime_profile": self.runtime_profile,
            "startup_state": self.startup_state,
            "mandatory_unresolved_contracts": self.mandatory_unresolved_contracts,
            "optional_skipped_contracts": self.optional_skipped_contracts,
            "ineligible_skipped_contracts": self.ineligible_skipped_contracts,
            "async_unsafe_handler_count": self.total_quarantined,
            "structural_invalid_handler_count": self.structural_invalid_handler_count,
            "failed_contract_count": self.total_failed,
        }

    def __bool__(self) -> bool:
        """True when all contracts wired or skipped (no failures).

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            there are zero failures. Differs from typical Pydantic behavior.
        """
        return self.total_failed == 0
