# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What became of one deploy job's reading row (OMN-24617)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelDeployJobReadingResult(BaseModel):
    """The row built for a job and whether the ledger took it.

    ``outcome`` is ``skipped`` when the job has no reading (a lane the reading does not
    cover, or a job with no terminal outcome), ``appended`` when the append command
    accepted the row, and ``unappended`` when it could not be written; ``detail`` then
    says why. ``row`` is empty when skipped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Literal["skipped", "appended", "unappended"]
    row: str = ""
    request_id: str = ""
    detail: str = ""
