# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Retry bounds of the auto-wiring consume boundary and NOT_READY reattach.

The two attempt counts the auto-wiring runtime uses are policy, not code: the
consume boundary retries a transient dispatch failure before routing the record
to the DLQ (OMN-14507), and the background reconciler re-attaches NOT_READY
contracts a bounded number of times (OMN-15215). Their defaults live here, on
one frozen policy model, so ``handler_wiring`` reads them instead of writing
the numbers as module literals.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelAutoWiringRetryPolicy"]


class ModelAutoWiringRetryPolicy(BaseModel):
    """Attempt bounds read by the auto-wiring runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    boundary_dlq_max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Dispatch attempts per consumed record at the consume boundary "
            "before a transient failure is routed to the DLQ (OMN-14507)."
        ),
    )
    not_ready_reattach_max_attempts: int = Field(
        default=5,
        ge=1,
        description=(
            "Background re-attach rounds for NOT_READY contracts after boot "
            "(OMN-15215)."
        ),
    )
