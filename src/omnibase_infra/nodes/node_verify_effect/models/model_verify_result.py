# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelVerifyResult — result from the verify effect node.

Related:
    - OMN-7317: node_verify_effect
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_verify_effect.models.model_verify_check import (
    ModelVerifyCheck,
)


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


__all__: list[str] = ["ModelVerifyResult"]
