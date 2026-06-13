# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declarative ``onex skill`` CLI-argument spec (OMN-13097).

One arg of an ``onex skill`` invocation, mapping a ``--flag`` (or positional)
onto a field of the backing node's contract input model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from omnibase_infra.cli.enum_skill_arg_type import EnumSkillArgType

__all__ = ["ModelSkillArgSpec"]


class ModelSkillArgSpec(BaseModel):
    """One CLI argument of an ``onex skill`` invocation.

    ``boolean`` args are presence flags (``--dry-run``); all others take a
    value (``--repos a,b,c``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="CLI flag name without leading dashes (e.g. 'repos', 'dry-run').",
    )
    payload_field: str = Field(
        ...,
        min_length=1,
        description="Field name in the backing node's input-model payload.",
    )
    arg_type: EnumSkillArgType = Field(
        ...,
        description="Coercion target for the raw CLI string value.",
    )
    required: bool = Field(
        default=False,
        description="Whether the argument must be supplied (no default applies).",
    )
    default: JsonValue = Field(
        default=None,
        description=(
            "Default payload value when the arg is omitted. Ignored when "
            "'required' is True. None means the field is omitted entirely "
            "(the node-input model supplies its own default)."
        ),
    )
    positional: bool = Field(
        default=False,
        description=(
            "When True the arg is read from trailing positional tokens "
            "(joined with spaces) rather than a --flag. At most one "
            "positional arg per skill."
        ),
    )

    @model_validator(mode="after")
    def _validate_required_has_no_default(self) -> ModelSkillArgSpec:
        if self.required and self.default is not None:
            raise ValueError(
                f"arg '{self.name}': required args must not declare a default"
            )
        return self
