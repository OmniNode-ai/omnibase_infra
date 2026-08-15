# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hash-chained durable cutover journal event."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.migration.cutover.models.model_cutover_journal_request import (
    ModelCutoverJournalRequest,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelCutoverJournalEvent(BaseModel):
    """Hash-chained durable event returned by the repository."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID
    family_id: UUID
    sequence: int = Field(..., ge=1)
    previous_event_hash: str = Field(..., pattern=_SHA256_PATTERN)
    event_hash: str = Field(..., pattern=_SHA256_PATTERN)
    request: ModelCutoverJournalRequest


__all__ = ["ModelCutoverJournalEvent"]
