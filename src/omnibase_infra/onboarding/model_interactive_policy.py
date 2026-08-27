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

    policy_name: str  # ONEX_EXCLUDE: pattern_validator - policy_name is the policy's own identifier, not an entity reference
    description: str
    version: dict[str, int]
    policy_type: Literal["interactive"]
    target_capabilities: list[str]
    max_estimated_minutes: int
    steps: list[ModelInteractiveStep]
    transitions: list[ModelTransition]
    env_output: dict[str, dict[str, str]]
    credentials_output: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description=(
            "Per-terminal-step secret material, as {terminal_step: "
            "{secret_ref_template: value_template}}. Kept separate from "
            "env_output rather than filtered out of it by key name "
            "(OMN-16038): env_output is rendered to the operator, written to "
            "the overlay, and echoed in receipts, so anything placed there is "
            "public by construction. A declared block is also the only reason "
            "the 0600 credentials artifact is ever written — an absent block "
            "means the policy handles no secrets at all."
        ),
    )
    start_step: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_graph_integrity(self) -> ModelInteractivePolicy:
        step_ids = [s.id for s in self.steps]
        if not step_ids:
            raise ValueError("Interactive policy must define at least one step")

        step_id_set = set(step_ids)

        if len(step_id_set) != len(step_ids):
            raise ValueError("Duplicate step IDs found")

        if self.start_step is None:
            object.__setattr__(self, "start_step", step_ids[0])
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

        # A credentials_output block keyed on a non-terminal step would never
        # fire — the reducer only emits credentials at a terminal step — and
        # the failure mode is a silently unwritten secret, so reject it here.
        for cid in self.credentials_output:
            if cid not in terminal_steps:
                raise ValueError(
                    f"credentials_output references '{cid}', which is not a "
                    f"terminal step; credentials are only emitted at terminal steps"
                )

        return self


__all__ = ["ModelInteractivePolicy"]
