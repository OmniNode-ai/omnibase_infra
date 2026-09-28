# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""A rendered proof run: ordered argv steps plus the identity they prove.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_kind import (
    EnumLabProofKind,
)
from omnibase_infra.lab_proof.model_lab_proof_step import ModelLabProofStep
from omnibase_infra.lab_proof.model_lab_proof_subject import ModelLabProofSubject


class ModelLabProofPlan(BaseModel):
    """The plan the run effect executes and the verdict handler reads back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_key: str
    host: str
    profile_key: str
    profile_version: int
    variant_key: str
    proof_kind: EnumLabProofKind
    subject: ModelLabProofSubject
    infra_sha: str
    workdir: str
    log_dir: str
    negative_control: bool
    steps: tuple[ModelLabProofStep, ...] = Field(min_length=1)


__all__ = ["ModelLabProofPlan"]
