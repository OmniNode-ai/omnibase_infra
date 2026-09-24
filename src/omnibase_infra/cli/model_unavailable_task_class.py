# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A task class the contract declares but cannot route yet (OMN-13966)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_routing_availability_status import (
    EnumRoutingAvailabilityStatus,
)

__all__ = ["ModelUnavailableTaskClass"]


class ModelUnavailableTaskClass(BaseModel):
    """One class carrying a ``routing_availability`` block, in the contract's words.

    The contract declares such a class so a consumer can refuse it up front
    instead of dispatching it and waiting out the ingress budget (OMN-16811).
    Every field here is copied from that block; none is written by the CLI.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    status: EnumRoutingAvailabilityStatus
    missing_capability: str = Field(min_length=1)
    tracking: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    def refusal(self) -> str:
        """Render the refusal an explicit ``--task-type`` naming this class gets."""
        return (
            f"task class {self.name!r} is declared by the task-class contract "
            f"but not routable: routing_availability status {self.status.value!r}, "
            f"missing capability {self.missing_capability!r} (tracking: "
            f"{self.tracking}). {self.reason}"
        )
