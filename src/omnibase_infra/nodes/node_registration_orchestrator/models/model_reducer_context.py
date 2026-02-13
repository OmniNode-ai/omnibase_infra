# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reducer Context Model.

Frozen Pydantic model that bundles tracing and timing parameters
common across all ``decide_*`` methods on RegistrationReducerService.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelReducerContext(BaseModel):
    """Context passed to reducer decision methods.

    Bundles tracing and timing parameters that are common across
    all reducer decision calls, reducing parameter counts on the
    ``decide_*`` methods.

    Attributes:
        correlation_id: Correlation ID for distributed tracing.
        now: Current time (injected, not generated).
        tick_id: UUID of the RuntimeTick that triggered this check (optional).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    now: datetime = Field(
        ...,
        description="Current time (injected, not generated)",
    )
    tick_id: UUID | None = Field(
        default=None,
        description="UUID of the RuntimeTick that triggered this check",
    )


__all__: list[str] = ["ModelReducerContext"]
