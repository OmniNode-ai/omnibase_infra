# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model for an interactive onboarding transition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.onboarding.model_transition_branch import ModelTransitionBranch


class ModelTransition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    from_step: str = Field(alias="from")
    terminal: bool = Field(default=False)
    responses: dict[str, ModelTransitionBranch] | None = Field(default=None)
    on_submit: list[ModelTransitionBranch] | None = Field(default=None)


__all__ = ["ModelTransition"]
