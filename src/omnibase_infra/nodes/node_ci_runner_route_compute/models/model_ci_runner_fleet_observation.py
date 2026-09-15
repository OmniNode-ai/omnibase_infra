# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""A capacity reading of the self-hosted runner group.

Ticket: OMN-18412
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCIRunnerFleetObservation(BaseModel):
    """A capacity reading of the self-hosted runner group.

    ``online``/``busy`` are optional because a failed probe has neither, and a
    zero would be a lie about a fleet nobody managed to read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ok: bool = Field(description="Whether the capacity probe succeeded.")
    error: str = Field(
        default="",
        description="Named failure class when ok is false (missing_token, timeout, "
        "http_403, malformed_json, empty_fleet, unexpected_shape).",
    )
    online: int | None = Field(
        default=None, ge=0, description="Runners online in the group."
    )
    busy: int | None = Field(
        default=None, ge=0, description="Runners currently executing a job."
    )


__all__ = ["ModelCIRunnerFleetObservation"]
