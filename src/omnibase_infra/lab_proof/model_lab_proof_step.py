# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""One planned step: an argv, never shell text.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_attribution import (
    EnumLabProofAttribution,
)
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.enum_lab_proof_step_phase import EnumLabProofStepPhase
from omnibase_infra.lab_proof.model_lab_proof_retry import ModelLabProofRetry


class ModelLabProofStep(BaseModel):
    """A step the run effect executes exactly as planned.

    The effect judges nothing: it runs ``argv`` in ``cwd``, applies the
    mechanical expectations declared here (exit status, an exact stdout, a
    non-empty stdout), counts ``grep_patterns``, and records what happened.
    What the observations mean is the verdict handler's job.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: EnumLabProofStepId
    phase: EnumLabProofStepPhase
    attribution: EnumLabProofAttribution
    purpose: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(ge=1)
    must_succeed: bool
    retry: ModelLabProofRetry | None = None
    expect_stdout_equals: str | None = None
    expect_stdout_empty: bool = False
    expect_stdout_nonempty: bool = False
    grep_patterns: tuple[str, ...] = ()
    extract_pattern: str = Field(
        default="",
        description="A regex with one group; every distinct capture in the output is "
        "recorded (sorted), so a verdict can compare WHICH items failed, not only "
        "how many lines matched.",
    )
    record_output: bool = Field(
        default=True,
        description="False keeps the output out of the report (it stays in the "
        "host log file): container logs can carry connection strings.",
    )


__all__ = ["ModelLabProofStep"]
