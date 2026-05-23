# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Input model for summarized invariant evaluation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.invariant import ModelInvariantSet


class ModelInvariantEvaluateAllInput(BaseModel):
    """Request for evaluating an invariant set with summary aggregation."""

    model_config = ConfigDict(frozen=True)

    invariant_set: ModelInvariantSet = Field(
        description="Invariant set to evaluate.",
    )
    output: dict[str, object] = Field(
        description="Execution output dictionary to validate against invariants.",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop on first critical or fatal failure when true.",
    )
    allowed_import_paths: list[str] | None = Field(
        default=None,
        description="Optional allow-list for custom invariant callable imports.",
    )


__all__ = ["ModelInvariantEvaluateAllInput"]
