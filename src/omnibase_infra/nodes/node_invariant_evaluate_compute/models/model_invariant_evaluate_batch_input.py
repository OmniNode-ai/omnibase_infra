# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Input model for batch invariant evaluation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.invariant import ModelInvariantSet


class ModelInvariantEvaluateBatchInput(BaseModel):
    """Request for evaluating an invariant set without summary aggregation."""

    model_config = ConfigDict(frozen=True)

    invariant_set: ModelInvariantSet = Field(
        description="Invariant set to evaluate.",
    )
    output: dict[str, object] = Field(
        description="Execution output dictionary to validate against invariants.",
    )
    enabled_only: bool = Field(
        default=True,
        description="Only evaluate enabled invariants when true.",
    )
    allowed_import_paths: list[str] | None = Field(
        default=None,
        description="Optional allow-list for custom invariant callable imports.",
    )


__all__ = ["ModelInvariantEvaluateBatchInput"]
