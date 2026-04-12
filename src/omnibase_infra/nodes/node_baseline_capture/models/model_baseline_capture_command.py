# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Command model for NodeBaselineCapture.

Ticket: OMN-7484
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelBaselineCaptureCommand(BaseModel):
    """Command to trigger a raw baseline measurement capture run.

    Attributes:
        operation: Fixed operation discriminator for handler routing.
        correlation_id: Required correlation ID for tracing (no default —
            callers must generate via ``uuid.uuid4()``).
        lookback_hours: How many hours of agent_actions to capture.
            Defaults to 24. Capped at 168 (7 days) by the handler.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["baselines.capture"] = "baselines.capture"
    correlation_id: UUID
    lookback_hours: int = 24


__all__: list[str] = ["ModelBaselineCaptureCommand"]
