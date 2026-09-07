# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Readiness-gate status naming the contracts that block ``/ready`` (OMN-17372).

This is the payload the ``contract_attach`` supplemental readiness probe puts
on ``/ready``'s ``supplemental_readiness`` block, so a 503 says WHICH contracts
are not attached rather than only that the runtime is not ready.

Related Tickets:
    - OMN-17372: readiness must require the wired command topics.
    - OMN-13237: the per-contract boot interleave that produces the results.
    - OMN-14758: the supplemental readiness probe seam this rides.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.runtime.enums.enum_contract_attach_gate_phase import (
    EnumContractAttachGatePhase,
)


class ModelContractAttachGateStatus(BaseModel):
    """Which required contracts are attached, and which block readiness.

    Attributes:
        phase: Lifecycle phase of the gate.
        ready: Whether every required contract reported ATTACHED.
        required_contracts: Contract names the gate requires, sorted.
        attached_contracts: Required contracts that reported ATTACHED, sorted.
        not_ready_contracts: Required contracts that reported NOT_READY, sorted.
        failed_contracts: Required contracts that reported FAILED, sorted.
        pending_contracts: Required contracts with no result yet, sorted. Any
            entry here means the boot interleave has not finished for that
            contract; the gate is NOT ready while this is non-empty.
        excluded_contracts: Contracts that subscribe a command topic but that
            the boot interleave reported it will NEVER attempt, each with the
            structural reason (OMN-17372). These are dropped from
            ``required_contracts`` — requiring a contract that can never report
            wedges ``/ready`` at 503 for the life of the process. A contract is
            only excluded while it has no NOT_READY / FAILED result of its own:
            a contract the interleave actually tried and could not attach stays
            blocking, whatever any later exclusion claims.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: EnumContractAttachGatePhase
    ready: bool
    required_contracts: tuple[str, ...] = Field(default_factory=tuple)
    attached_contracts: tuple[str, ...] = Field(default_factory=tuple)
    not_ready_contracts: tuple[str, ...] = Field(default_factory=tuple)
    failed_contracts: tuple[str, ...] = Field(default_factory=tuple)
    pending_contracts: tuple[str, ...] = Field(default_factory=tuple)
    excluded_contracts: tuple[ModelContractAttachExclusion, ...] = Field(
        default_factory=tuple
    )


__all__: list[str] = ["ModelContractAttachGateStatus"]
