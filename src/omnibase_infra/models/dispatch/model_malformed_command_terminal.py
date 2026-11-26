# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed failure terminal for a publisher-malformed command (OMN-26026)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelMalformedCommandTerminal"]


class ModelMalformedCommandTerminal(BaseModel):
    """Typed terminal for a publisher-malformed command."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = Field(..., min_length=1, description="Declared failure topic.")
    correlation_id: UUID = Field(..., description="Correlation of the refused command.")
    workflow_id: str | None = Field(
        default=None, description="Workflow id carried by the command, if any."
    )
    status: Literal["failed"] = Field(default="failed")
    failure_class: Literal["publisher_malformed"] = Field(default="publisher_malformed")
    failure_code: str | None = Field(default=None)
    retryable: Literal[False] = Field(default=False)
    failure_reason: str = Field(..., description="Sanitized validation detail.")
    origin_topic: str = Field(..., description="Command topic that was refused.")
