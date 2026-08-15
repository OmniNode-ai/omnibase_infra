# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregate per-contract attach readiness for the runtime (OMN-13237, §3.8/§3.10).

The readiness endpoint reports attach status ONLY — it is not a source of truth
for contract lifecycle.

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
    - OMN-15512: Ride the existing runtime-manifest event/projection so the
      NOT-READY blocker set is durably queryable instead of log-only.
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

    @property
    def not_ready_results(self) -> tuple[ModelContractAttachResult, ...]:
        """The blocker set: every contract whose consumer did NOT attach.

        Covers both ``NOT_READY`` (readiness confirm failed, attach skipped)
        and ``FAILED`` (attach raised after readiness passed). Order follows
        ``results``, which is boot-walk order.
        """
        return tuple(
            r for r in self.results if r.status is not EnumContractAttachStatus.ATTACHED
        )

    def blockers_only(self) -> ModelRuntimeAttachReadiness:
        """Return a copy whose ``results`` hold ONLY the blocker set.

        Used for the published/persisted copy (OMN-15512). The counts are
        preserved, so ``required_contracts - attached_contracts`` still equals
        ``len(results)`` on the narrowed copy and no information is lost:
        contracts that DID attach are already enumerated on the same
        runtime-manifest payload (``contracts`` / ``handlers`` / the
        subscribed-topic set). Re-emitting several hundred attached results
        would roughly double the envelope for zero added signal, and boot walks
        475+ contracts today.
        """
        return self.model_copy(update={"results": self.not_ready_results})


__all__: list[str] = ["ModelRuntimeAttachReadiness"]
