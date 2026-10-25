# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed options at the work-ledger Click command boundary."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelWorkLedgerQueryCliOptions(BaseModel):
    """Click parses the values; this model keeps its callback's input cohesive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bus_lane: Literal["dev"]
    kind: str
    since: datetime | None
    until: datetime | None
    lane: str | None
    ticket: str | None
    pr: str | None
    term: str | None
    kinds: tuple[str, ...]
    limit: int | None
    read_stdin: bool
    timeout_s: float = Field(gt=0)
