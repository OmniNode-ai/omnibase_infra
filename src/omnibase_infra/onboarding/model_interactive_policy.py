# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model for the interactive onboarding policy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep
from omnibase_infra.onboarding.model_transition import ModelTransition


class ModelInteractivePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_name: str
    description: str
    version: dict[str, int]
    policy_type: Literal["interactive"]
    target_capabilities: list[str]
    max_estimated_minutes: int
    steps: list[ModelInteractiveStep]
    transitions: list[ModelTransition]
    env_output: dict[str, dict[str, str]]
    start_step: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_graph_integrity(self) -> ModelInteractivePolicy:
        if self.start_step is None:
            object.__setattr__(self, "start_step", self.steps[0].id)

        step_ids = [s.id for s in self.steps]
        step_id_set = set(step_ids)

        if len(step_id_set) != len(step_ids):
            raise ValueError("Duplicate step IDs found")

        terminal_steps: set[str] = set()
        for t in self.transitions:
            if t.from_step not in step_id_set:
                raise ValueError(f"Transition references unknown step '{t.from_step}'")
            if t.terminal:
                terminal_steps.add(t.from_step)
                continue
            for branch in (t.responses or {}).values():
                if branch.next not in step_id_set:
                    raise ValueError(
                        f"Transition from '{t.from_step}' references unknown step '{branch.next}'"
                    )
            for branch in t.on_submit or []:
                if branch.next not in step_id_set:
                    raise ValueError(
                        f"Transition from '{t.from_step}' references unknown step '{branch.next}'"
                    )

        for tid in terminal_steps:
            if tid not in self.env_output:
                raise ValueError(f"Terminal step '{tid}' missing from env_output")

        return self


__all__ = ["ModelInteractivePolicy"]
