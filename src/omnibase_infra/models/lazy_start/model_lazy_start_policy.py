# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane overlay's ``lazy_start`` block (OMN-21592, plan task A4, section 3.11.5).

A lane whose hot-load row is ``enabled`` with ``mode: on_demand`` and carries this
block boots only the contracts it names: ``ingress_contract`` (the front door that
receives first requests) and ``boot_contracts`` (the boot-priority set that stays
eager). Every other contract the profile owns stays dormant: its handler class is
not imported, no dispatcher is registered, no topic is provisioned and no consumer
is attached until its first request reaches the front door. The names are data in
the overlay row; no contract name is spelled in code.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__: list[str] = ["ModelLazyStartPolicy"]


class ModelLazyStartPolicy(BaseModel):
    """Boot set, start budget and retry budget of one lane."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ingress_contract: str = Field(
        min_length=1,
        description="The front-door contract; always booted on a lazy-start lane",
    )
    boot_contracts: tuple[str, ...] = Field(
        default=(),
        description="Boot-priority contracts that stay eager (wired at boot)",
    )
    budget_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="A first request not answered inside this is START_TIMEOUT",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Further start attempts after a failed start, per request",
    )

    @model_validator(mode="after")
    def _names_are_unique_and_not_blank(self) -> Self:
        names = (self.ingress_contract, *self.boot_contracts)
        if any(not name.strip() for name in names):
            raise ValueError("lazy_start names must not be blank")
        if len(set(self.boot_contracts)) != len(self.boot_contracts):
            raise ValueError("lazy_start.boot_contracts must not repeat a name")
        return self

    @property
    def boot_set(self) -> frozenset[str]:
        """Every contract this lane wires at boot."""
        return frozenset((self.ingress_contract, *self.boot_contracts))
