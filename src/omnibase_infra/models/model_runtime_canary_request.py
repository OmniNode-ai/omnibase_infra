# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One runtime canary run: the candidate and the bounds it is held to (OMN-26463)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_runtime_canary_candidate import (
    ModelRuntimeCanaryCandidate,
)

__all__: list[str] = ["ModelRuntimeCanaryRequest"]


class ModelRuntimeCanaryRequest(BaseModel):
    """The ticket's bounds: /ready inside 5 minutes, the heartbeat healthy for 15."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: ModelRuntimeCanaryCandidate
    receipt_dir: Path
    ready_budget_seconds: float = Field(default=300.0, gt=0)
    heartbeat_hold_seconds: float = Field(default=900.0, gt=0)
    poll_seconds: float = Field(default=5.0, gt=0)
