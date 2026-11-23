# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The front door's serving set and wake failures, read from outside the process (OMN-26316).

The contract hot-loading plan's serving-set read slice: the contracts the front door
has woken (its ``serving`` set) and, per failed wake, the contract and the reason,
live only on the ``LazyStartFrontDoor`` object inside the running runtime. This is
the snapshot ``/ready`` carries, so a reader outside the process sees a dormant
contract attach on its first request.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__: list[str] = ["ModelLazyStartServingStatus"]


class ModelLazyStartServingStatus(BaseModel):
    """One read of the front door: dormant at boot, woken since, and failed wakes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dormant_contract_count: int = Field(
        ..., ge=0, description="Contracts left dormant by the boot split."
    )
    serving_contracts: tuple[str, ...] = Field(
        ..., description="Dormant contracts the front door woke, sorted by name."
    )
    serving_count: int = Field(..., ge=0, description="len(serving_contracts).")
    wake_failures: dict[str, str] = Field(
        ..., description="Contract name to the reason its last wake failed."
    )
