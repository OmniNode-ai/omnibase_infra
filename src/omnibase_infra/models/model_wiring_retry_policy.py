# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wiring retry policy model (OMN-18087)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelWiringRetryPolicy(BaseModel):
    """How many times the auto-wiring loop retries a consume or a NOT_READY attach.

    The counts are policy field defaults rather than module constants in the
    wiring loop, which holds no operational number of its own.

    Attributes:
        boundary_dlq_max_attempts: Dispatch attempts at the auto-wired consume
            boundary before the message is routed to the DLQ (OMN-14507).
        not_ready_attempts: Background re-attach attempts for a contract whose
            topics were NOT_READY at boot (OMN-15215).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    boundary_dlq_max_attempts: int = Field(default=3, ge=1)
    not_ready_attempts: int = Field(default=5, ge=1)


__all__: list[str] = ["ModelWiringRetryPolicy"]
