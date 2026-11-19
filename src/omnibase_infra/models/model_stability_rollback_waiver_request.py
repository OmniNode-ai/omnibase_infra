# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the deploy agent hands the stability rollback waiver when no point was retained (OMN-24562)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelStabilityRollbackWaiverRequest(BaseModel):
    """The deploy's lane and scope and the guard probe's reading of each runtime target.

    ``readings`` holds one ``(status, detail)`` pair per target, ``status`` being
    ``pass``, ``fail`` or ``timeout``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: str
    scope: str
    point_retained: bool
    readings: tuple[tuple[str, str], ...] = ()
