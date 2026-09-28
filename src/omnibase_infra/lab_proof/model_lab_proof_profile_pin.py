# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""What the registry says one repository's PR-head receipt is judged against.

A publisher of a ``pr-head`` receipt (the ``lab-proof-receipt`` check run, the
``lab-proof`` gate) pins the profile key, version and mandatory checks from the
reviewed registry rather than from its caller, so the only way to weaken a
verdict is a reviewed registry change.

Ticket: OMN-19566
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.enum_lab_proof_kind import EnumLabProofKind


class ModelLabProofProfilePin(BaseModel):
    """One repository's current profile and one variant's mandatory checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(pattern=r"^OmniNode-ai/[A-Za-z0-9_.-]+$")
    proof_kind: EnumLabProofKind
    profile_key: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_-]*$")
    profile_version: str = Field(pattern=r"^[1-9][0-9]*$")
    mandatory_checks: tuple[EnumLabProofCheck, ...] = Field(min_length=1)


__all__ = ["ModelLabProofProfilePin"]
