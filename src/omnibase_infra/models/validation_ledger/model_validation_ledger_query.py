# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation ledger query model.

This module defines the filter model used to query the validation_event_ledger
table. All filter fields are optional to support flexible query composition.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelValidationLedgerQuery(BaseModel):
    """Filter model for validation ledger queries. All fields optional."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    run_id: UUID | None = Field(default=None, description="Filter by validation run ID")
    repo_id: str | None = Field(default=None, description="Filter by repository ID")
    event_type: str | None = Field(default=None, description="Filter by event type")
    start_time: datetime | None = Field(
        default=None, description="Filter events after this time"
    )
    end_time: datetime | None = Field(
        default=None, description="Filter events before this time"
    )
    limit: int = Field(default=100, ge=1, le=10000, description="Max results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
