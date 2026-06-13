# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declarative ``onex skill`` → backing-node mapping model (OMN-13097).

Phase 4a of the skill-output-suppression slice
(``docs/plans/2026-06-12-skill-output-suppression-plan.md`` Phase 4 item 1).

``onex skill <name> [args]`` is the single-command dispatch surface that
replaces the 24 hand-written dispatch shims. The skill→node mapping is
DECLARATIVE DATA — it lives in ``skill_mapping.yaml`` beside this module and
is loaded into these frozen, typed models; it is NEVER hardcoded as Python
branching in the command (ticket deliverable 2). Each entry declares:

- the backing ONEX node (resolved via the ``onex.nodes`` entry-point group,
  exactly like ``onex node``),
- how the skill's CLI arguments map onto fields of the node's contract input
  model (name, type, default, required),
- optional keyword classifiers (the delegate skill's ``task_type``
  auto-classification, expressed as data rather than inline logic),
- static payload fields the node requires regardless of args.

.. versionadded:: OMN-13097
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from omnibase_infra.cli.model_skill_arg_spec import ModelSkillArgSpec
from omnibase_infra.cli.model_skill_classifier import ModelSkillClassifier

__all__ = ["ModelSkillMapping"]


class ModelSkillMapping(BaseModel):
    """One ``onex skill <name>`` → backing node mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_name: str = Field(
        ...,
        min_length=1,
        description="Skill name as invoked: 'onex skill <skill_name>'.",
    )
    node_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Backing node registered under the 'onex.nodes' entry-point group."
        ),
    )
    result_model: str = Field(
        ...,
        min_length=1,
        description=(
            "Fully qualified name of the backing node's typed handler result "
            "model — schema identity surfaced in the receipt (ticket "
            "deliverable 3)."
        ),
    )
    event_bus: str = Field(
        default="inmemory",
        description="event_bus backend override for the dispatch.",
    )
    timeout: int = Field(
        default=300,
        gt=0,
        description="Max dispatch execution time in seconds.",
    )
    args: tuple[ModelSkillArgSpec, ...] = Field(
        default=(),
        description="Declarative CLI-arg → payload-field specs.",
    )
    static_payload: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Payload fields injected regardless of supplied args.",
    )
    classifiers: tuple[ModelSkillClassifier, ...] = Field(
        default=(),
        description="Keyword classifiers applied to still-unset fields.",
    )

    @model_validator(mode="after")
    def _validate_single_positional(self) -> ModelSkillMapping:
        positionals = [a for a in self.args if a.positional]
        if len(positionals) > 1:
            raise ValueError(
                f"skill '{self.skill_name}': at most one positional arg allowed, "
                f"got {len(positionals)}"
            )
        return self
