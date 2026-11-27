# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which queued requests may fold, and whether an operator-path request runs (OMN-26718)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelQueuedRequestPriorityDecision(BaseModel):
    """The head and the next ``foldable - 1`` entries pass every static rule.

    ``refusal`` names why the entry at position ``foldable`` does not (empty when none
    follows). When ``operator_path`` is true the entry at ``runner_index`` is an
    operator-path request that runs, ``foldable == runner_index + 1``, and ``reasons[i]`` is
    the recorded reason for the automatic request at position ``i``, naming the request
    that runs in its place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    foldable: int = Field(ge=1)
    refusal: str = ""
    operator_path: bool = False
    runner_index: int = Field(default=0, ge=0)
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _operator_path_is_consistent(self) -> ModelQueuedRequestPriorityDecision:
        if self.operator_path and (
            self.foldable != self.runner_index + 1
            or len(self.reasons) != self.runner_index
        ):
            raise ValueError(
                "an operator-path decision runs the last foldable entry and carries "
                "one reason per superseded entry"
            )
        if not self.operator_path and (self.runner_index or self.reasons):
            raise ValueError("a runner and reasons belong to an operator-path decision")
        return self
