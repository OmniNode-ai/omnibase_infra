# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one operation on the dev deploy agent's code-ref pin (OMN-23039)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_dev_lane_code_ref_pin import ModelDevLaneCodeRefPin


class ModelDevLaneCodeRefPinResult(BaseModel):
    """The record file touched and, for ``active``, the live pin (None when none)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["register", "active", "release"]
    path: Path
    pin: ModelDevLaneCodeRefPin | None = None
