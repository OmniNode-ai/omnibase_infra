# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A deploy job's settled record, as the reading of its terminal outcome reads it (OMN-24617)."""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class ModelDeployJobReadingRequest(BaseModel):
    """The facts of one job the deploy agent settled, and the time its reading is read.

    ``phase_results`` maps each deploy phase name to its settled status; ``not_started``
    is the reason no phase ran, empty when one did; ``first_error`` is the first line of the
    job's recorded errors, empty when it recorded none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str = Field(min_length=1)
    runtime_lane: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    status: str = Field(min_length=1)
    phase_results: dict[str, str] = Field(default_factory=dict)
    not_started: str = ""
    first_error: str = ""
    completed_at: AwareDatetime | None = None
    published_at: AwareDatetime
