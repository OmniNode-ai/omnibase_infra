# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Checkout and checkin counters a handler pool holds outside the pool (OMN-21536)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPoolCheckoutMetrics(BaseModel):
    """Running totals of the checkouts a pool served.

    The pool hands this collaborator the measurements; the pool itself holds no
    counter, so its public checkout method does not mutate handler state.
    """

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    checkout_count: int = Field(default=0, ge=0, description="Checkouts served.")
    checkin_count: int = Field(default=0, ge=0, description="Instances returned.")
    wait_total_ms: float = Field(
        default=0.0, ge=0, description="Summed milliseconds callers waited."
    )

    def record_checkout(self, wait_ms: float) -> None:
        """Count one checkout that waited ``wait_ms`` for an instance."""
        self.checkout_count += 1
        self.wait_total_ms += wait_ms

    def record_checkin(self) -> None:
        """Count one instance returned to the pool."""
        self.checkin_count += 1

    @property
    def average_wait_ms(self) -> float:
        """Mean wait per checkout, zero before the first one."""
        return self.wait_total_ms / self.checkout_count if self.checkout_count else 0.0


__all__: list[str] = ["ModelPoolCheckoutMetrics"]
