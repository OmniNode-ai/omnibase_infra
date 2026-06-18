# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Workspace pre-flight validate result (OMN-13247, plan §5.5).

The deterministic verdict the COMPUTE node returns for a
``ModelWorkspaceValidateCommand``. The orchestrator gates whether any subprocess
runs on ``valid``.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelWorkspaceValidateResult(BaseModel):
    """Deterministic verdict of the workspace pre-flight check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow run correlation id.")
    valid: bool = Field(..., description="True iff the workspace passed all checks.")
    resolved_path: str = Field(
        default="", description="The symlink-resolved absolute path."
    )
    rejection_reason: str | None = Field(
        default=None, description="Why validation failed; None if valid."
    )

    def __bool__(self) -> bool:
        """Allow ``if result:`` to mean "validation passed".

        Warning:
            Non-standard ``__bool__``: returns ``valid`` rather than the default
            truthiness of a populated model.
        """
        return self.valid


__all__: list[str] = ["ModelWorkspaceValidateResult"]
