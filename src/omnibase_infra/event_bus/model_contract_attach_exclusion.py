# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract the boot interleave structurally will not attempt (OMN-17372).

The sibling of
:class:`~omnibase_infra.event_bus.model_contract_attach_result.ModelContractAttachResult`:
that model says what happened when the interleave TRIED a contract, this one
says the interleave will never try it, and why. Between them the interleave
accounts for every contract in the wiring report, so a consumer can tell
"has not reported yet" apart from "will never report" instead of waiting
forever on the second while believing it is the first.

Related Tickets:
    - OMN-17372: the readiness gate required contracts the interleave never
      attempts, so ``/ready`` stayed 503 for the life of the process.
    - OMN-13237: the per-contract provision -> confirm-ready -> attach interleave.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)


class ModelContractAttachExclusion(BaseModel):
    """One contract the interleave filtered out before attempting attach.

    Attributes:
        contract_name: The contract's node name, as it appears in the manifest
            and in the readiness gate's required set.
        reason: Which structural filter dropped it.
        detail: Human-readable detail (no secrets) — for ``NOT_WIRED`` this
            carries the wiring result's own ``reason`` string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_name: str
    reason: EnumContractAttachExclusionReason
    detail: str = Field(default="")


__all__: list[str] = ["ModelContractAttachExclusion"]
