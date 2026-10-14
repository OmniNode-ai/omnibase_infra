# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request and result models of the stand-in sweep and its stages (OMN-21071)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelStandInSweepRequest(BaseModel):
    """One sweep: the operand every stage answers for, and how long to wait."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    operand: str
    stage_deadline_s: float = Field(gt=0)


class ModelStandInSweepResult(BaseModel):
    """The sweep's own terminal: each stage's answer, or the stage that never answered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    status: str
    stage_values: dict[str, str] = Field(default_factory=dict)
    ignored_terminals: int = 0
    why: str = ""


class ModelStandInStageRequest(BaseModel):
    """One stage call, keyed by the sweep's correlation id and the graph node id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    stage_key: str
    operand: str


class ModelStandInStageResult(BaseModel):
    """One stage's terminal, carrying the key it answers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    stage_key: str
    value: str
