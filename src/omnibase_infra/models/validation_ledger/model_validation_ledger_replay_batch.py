# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation ledger replay batch model.

This module defines the paginated result set returned from a validation
ledger query, containing matched entries and pagination metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation_ledger.model_validation_ledger_entry import (
    ModelValidationLedgerEntry,
)
from omnibase_infra.models.validation_ledger.model_validation_ledger_query import (
    ModelValidationLedgerQuery,
)


class ModelValidationLedgerReplayBatch(BaseModel):
    """Paginated result set from a validation ledger query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[ModelValidationLedgerEntry, ...] = Field(default_factory=tuple)
    total_count: int = Field(..., ge=0, description="Total matching entries")
    has_more: bool = Field(
        ..., description="Whether more entries exist beyond this page"
    )
    query: ModelValidationLedgerQuery = Field(
        ..., description="Original query for reference"
    )
