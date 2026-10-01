# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Where an ``onex skill`` resolves a task class from (OMN-19407).

This replaced the keyword classifier (``model_skill_classifier.py``), whose
rules and ``research`` fallback were the same hand-written table OMN-18305
removed from ``onex delegate`` and which survived here. A mapping now only
names the two payload fields involved; the class itself is resolved by the
task-class contract read through the registry, exactly as ``onex delegate``
resolves it, so the two commands cannot disagree on a prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelSkillTaskClassResolution"]


class ModelSkillTaskClassResolution(BaseModel):
    """The payload fields a task class is resolved into and from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_field: str = Field(
        ...,
        min_length=1,
        description=(
            "Payload field holding the task class. An explicit value is checked "
            "against the task-class contract; an unset one is resolved by it."
        ),
    )
    prompt_field: str = Field(
        ...,
        min_length=1,
        description="Payload field whose text the contract's selection reads.",
    )
