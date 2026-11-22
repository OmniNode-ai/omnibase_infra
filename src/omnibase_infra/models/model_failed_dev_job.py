# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelFailedDevJob(BaseModel):
    """The newest terminal dev job when it failed, and the source it built."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    requested_by: str
    #: Repository -> the commit the failed job built, as its record names it.
    sources: dict[str, str]
