# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mark stale intent payload model.

Related:
    - OMN-1653: Contract Registry Reducer implementation
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadMarkStale(BaseModel):
    """Payload for PostgreSQL mark stale intents.

    Used when a runtime-tick event is processed to batch mark
    contracts as stale if they haven't been seen within the threshold.

    Attributes:
        intent_type: Routing discriminator. Always "postgres.mark_stale".
        correlation_id: Correlation ID for distributed tracing.
        stale_cutoff: Contracts with last_seen_at before this are stale.
        checked_at: When staleness check was performed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["postgres.mark_stale"] = Field(
        default="postgres.mark_stale",
        description="Routing discriminator for intent dispatch.",
    )

    correlation_id: UUID = Field(
        ..., description="Correlation ID for distributed tracing."
    )

    stale_cutoff: datetime = Field(
        ..., description="Contracts older than this are marked stale."
    )

    checked_at: datetime = Field(..., description="When check was performed.")


__all__ = ["ModelPayloadMarkStale"]
