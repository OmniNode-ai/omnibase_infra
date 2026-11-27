# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which auto-wired contracts the runtime joins at boot (OMN-18087)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelBootSubscriptionsConfig(BaseModel):
    """Boot subscription policy, declared in the runtime config contract.

    Attributes:
        policy: ``all`` joins every auto-wired contract's consumer groups at
            boot. ``in_use`` joins at boot only the contracts with a subscribe
            topic holding a retained record, as the broker reports it, and
            joins each of the rest when its first request arrives.
            ``gate_required`` (OMN-25856) joins at boot only the priority
            contracts, the command contracts in use (the ones the contract-attach
            readiness gate requires) and any contract with a topic that cannot be
            watched; every other contract, one holding a retained record on a
            non-command topic included, joins when the next record reaches one of
            its topics, from ``earliest``.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    policy: Literal["all", "in_use", "gate_required"] = Field(
        default="all",
        description="Which auto-wired contracts join their consumer groups at boot",
    )

    @property
    def demand_start(self) -> bool:
        """Whether contracts the policy defers start on their first record."""
        return self.policy != "all"

    @property
    def defer_retained(self) -> bool:
        """Whether a retained non-command topic also defers its contract."""
        return self.policy == "gate_required"


__all__: list[str] = ["ModelBootSubscriptionsConfig"]
