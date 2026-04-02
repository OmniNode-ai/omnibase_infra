# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verify effect models.

Related:
    - OMN-7317: node_verify_effect
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelVerifyCheck(BaseModel):
    """Result of a single verification check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Check name.")
    passed: bool = Field(..., description="Whether the check passed.")
    critical: bool = Field(
        default=True, description="Whether failure is critical (blocks loop) or just a warning."
    )
    message: str = Field(default="", description="Details about the check result.")


class ModelVerifyInput(BaseModel):
    """Input to the verify effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Build loop cycle correlation ID.")
    dry_run: bool = Field(default=False, description="Skip actual checks.")


class ModelVerifyResult(BaseModel):
    """Result from the verify effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Build loop cycle correlation ID.")
    all_critical_passed: bool = Field(
        ..., description="Whether all critical checks passed."
    )
    checks: tuple[ModelVerifyCheck, ...] = Field(
        ..., description="Individual check results."
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple, description="Non-critical warnings."
    )


__all__: list[str] = ["ModelVerifyCheck", "ModelVerifyInput", "ModelVerifyResult"]
