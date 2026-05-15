# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model for the interactive onboarding policy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.onboarding.enum_interactive_policy_type import (
    EnumInteractivePolicyType,
)
from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep
from omnibase_infra.onboarding.model_transition import ModelTransition


class ModelInteractivePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_slug: str = Field(alias="policy_name")
    display_name: str | None = Field(default=None)
    description: str
    version: dict[str, int]
    policy_type: EnumInteractivePolicyType
    target_capabilities: list[str]
    max_estimated_minutes: int
    steps: list[ModelInteractiveStep]
    transitions: list[ModelTransition]
    env_output: dict[str, dict[str, str]]
    start_step: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_graph_integrity(self) -> ModelInteractivePolicy:
        step_ids_list = [s.id for s in self.steps]
        if not step_ids_list:
            raise ValueError("Interactive policy must define at least one step")

        step_id_set = set(step_ids_list)

        if len(step_id_set) != len(step_ids_list):
            raise ValueError("Duplicate step IDs found")

        if self.start_step is None:
            object.__setattr__(self, "start_step", step_ids_list[0])
        elif self.start_step not in step_id_set:
            raise ValueError(f"start_step references unknown step '{self.start_step}'")

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

    @property
    def policy_name(self) -> str:
        return self.policy_slug


__all__ = ["ModelInteractivePolicy"]
