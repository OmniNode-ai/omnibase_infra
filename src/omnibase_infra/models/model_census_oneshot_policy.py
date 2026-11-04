# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Composition-root policy supplied from the node contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCensusOneshotPolicy(BaseModel):
    """Composition-root policy supplied from the node contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    phase_timeouts: dict[str, int]
    sample_interval_seconds: int = Field(gt=0)
    dispatch_timeout_seconds: int = Field(gt=0)
