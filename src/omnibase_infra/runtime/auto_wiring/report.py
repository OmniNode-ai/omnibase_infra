# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wiring report models for OMN-7654.

Captures per-contract wiring outcomes: success, skip, or failure with reason.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EnumWiringOutcome(str, Enum):
    """Outcome of wiring a single discovered contract."""

    WIRED = "wired"
    SKIPPED = "skipped"
    FAILED = "failed"


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
