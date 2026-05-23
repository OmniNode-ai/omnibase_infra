# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Input model for single invariant evaluation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.invariant import ModelInvariant


class ModelInvariantEvaluateInput(BaseModel):
    """Request for evaluating one invariant."""

    model_config = ConfigDict(frozen=True)

    invariant: ModelInvariant = Field(description="Invariant to evaluate.")
    output: dict[str, object] = Field(
        description="Execution output dictionary to validate against an invariant.",
    )
    allowed_import_paths: list[str] | None = Field(
        default=None,
        description="Optional allow-list for custom invariant callable imports.",
    )


__all__ = ["ModelInvariantEvaluateInput"]
