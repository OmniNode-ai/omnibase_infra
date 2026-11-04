# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Requests for census sampling and pure oneshot grading."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelCensusOneshotPolicy(BaseModel):
    """Composition-root policy supplied from the node contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    phase_timeouts: dict[str, int]
    sample_interval_seconds: int = Field(gt=0)
    dispatch_timeout_seconds: int = Field(gt=0)


class ModelCensusOneshotRequest(BaseModel):
    """One bounded sampling invocation of the existing collector."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    operation: Literal["census_oneshot.sample"] = "census_oneshot.sample"
    policy: ModelCensusOneshotPolicy
    script_directory: str
    arguments: tuple[str, ...] = ()


class ModelOneshotObservation(BaseModel):
    """Only the fields needed to grade one run-to-completion execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    completion_timeout_seconds: int = Field(gt=0)
    lane: str
    container: str
    keepalive: bool = False
    present: bool = True
    running: bool = False
    status: str = ""
    started_at: str = ""
    observed_at: datetime | None = None
