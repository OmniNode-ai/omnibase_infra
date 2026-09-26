# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The verdict of one proof run, keyed by repo, PR, head, profile and version.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.enum_lab_proof_outcome import EnumLabProofOutcome
from omnibase_infra.lab_proof.model_lab_proof_check_result import (
    ModelLabProofCheckResult,
)


class ModelLabProofResult(BaseModel):
    """What a prover reports and what the pr-head receipt (OMN-19566) will carry.

    ``receipt_key`` is the plan's key (section 3):
    ``<repo>#<pr>@<head_sha>:<profile_key>@<profile_version>``. A base control
    run and a negative control run carry the same key and say so in
    ``proved_sha`` and ``negative_control``; neither is ever a PASS for the head.
    ``restored`` is the zero-residue readback, reported beside the outcome and
    never folded into it: a dirty host is a host fact, not a fact about the PR.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt_key: str
    repo: str
    pr_number: int
    head_sha: str
    base_sha: str
    proved_sha: str
    profile_key: str
    profile_version: int
    variant_key: str
    host: str
    run_key: str
    negative_control: bool
    base_control: bool
    outcome: EnumLabProofOutcome
    checks: tuple[ModelLabProofCheckResult, ...]
    reasons: tuple[str, ...] = ()
    restored: bool
    residue_detail: tuple[str, ...] = ()
    started_at: str
    finished_at: str
    failed_checks: tuple[EnumLabProofCheck, ...] = Field(default=())


__all__ = ["ModelLabProofResult"]
