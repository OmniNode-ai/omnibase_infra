# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregate per-contract attach readiness for the runtime (OMN-13237, §3.8/§3.10).

The readiness endpoint reports attach status ONLY — it is not a source of truth
for contract lifecycle.

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.enum_runtime_readiness_state import (
    EnumRuntimeReadinessState,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)


class ModelRuntimeAttachReadiness(BaseModel):
    """Aggregate per-contract attach readiness for the runtime (§3.8, §3.10).

    Attributes:
        state: Aggregate tri-state (ready/degraded/failed).
        required_contracts: Count of contracts that should attach.
        attached_contracts: Count of contracts whose consumer attached.
        results: Per-contract attach results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: EnumRuntimeReadinessState = Field(default=EnumRuntimeReadinessState.READY)
    required_contracts: int = Field(default=0, ge=0)
    attached_contracts: int = Field(default=0, ge=0)
    results: tuple[ModelContractAttachResult, ...] = Field(default_factory=tuple)

    @classmethod
    def from_results(
        cls,
        results: tuple[ModelContractAttachResult, ...],
        *,
        core_contract_names: frozenset[str] = frozenset(),
    ) -> ModelRuntimeAttachReadiness:
        """Aggregate per-contract results into the runtime tri-state.

        A core control-plane contract that did not attach yields ``FAILED``;
        any other not-attached contract yields ``DEGRADED``; all attached
        yields ``READY``. Liveness is never derived from this aggregate.
        """
        required = len(results)
        attached = sum(
            1 for r in results if r.status is EnumContractAttachStatus.ATTACHED
        )
        core_gap = any(
            r.contract_name in core_contract_names
            and r.status is not EnumContractAttachStatus.ATTACHED
            for r in results
        )
        if core_gap:
            state = EnumRuntimeReadinessState.FAILED
        elif attached < required:
            state = EnumRuntimeReadinessState.DEGRADED
        else:
            state = EnumRuntimeReadinessState.READY
        return cls(
            state=state,
            required_contracts=required,
            attached_contracts=attached,
            results=results,
        )


__all__: list[str] = ["ModelRuntimeAttachReadiness"]
