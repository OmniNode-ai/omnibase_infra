# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The alarm row of one routed command with no job record (OMN-26485)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelRoutedCommandAlarmRow(BaseModel):
    """The row built for a command and whether the ledger took it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    instance: str
    row: str
    append_key: str
    outcome: Literal["appended", "unappended"]
    detail: str = ""
