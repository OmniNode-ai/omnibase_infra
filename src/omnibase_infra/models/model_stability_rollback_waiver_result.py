# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability rollback waiver's decision (OMN-24562)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelStabilityRollbackWaiverResult(BaseModel):
    """Whether the deploy may run, how the lane reads and, when refused, why.

    ``waived`` is true only when the deploy runs with no retained rollback point.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    admitted: bool
    waived: bool = False
    reading: str = ""
    detail: str = ""
    refusal: str = ""
