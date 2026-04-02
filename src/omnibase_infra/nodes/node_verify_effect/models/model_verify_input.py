# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelVerifyInput — input to the verify effect node.

Related:
    - OMN-7317: node_verify_effect
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelVerifyInput(BaseModel):
    """Input to the verify effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Build loop cycle correlation ID.")
    dry_run: bool = Field(default=False, description="Skip actual checks.")


__all__: list[str] = ["ModelVerifyInput"]
