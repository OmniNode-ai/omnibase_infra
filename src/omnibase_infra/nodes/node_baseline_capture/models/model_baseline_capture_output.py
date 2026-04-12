# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Output model for NodeBaselineCapture.

Ticket: OMN-7484
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelBaselineCaptureOutput(BaseModel):
    """Result of a raw baseline measurement capture run.

    Attributes:
        measurements_captured: Number of agent_actions rows read.
        snapshot_emitted: True when the baselines-computed event was published.
        errors: Tuple of sanitized error messages (empty on success).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    measurements_captured: int
    snapshot_emitted: bool
    errors: tuple[str, ...] = ()


__all__: list[str] = ["ModelBaselineCaptureOutput"]
