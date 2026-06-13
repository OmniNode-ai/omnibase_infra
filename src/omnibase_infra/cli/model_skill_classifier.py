# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declarative ``onex skill`` keyword classifier (OMN-13097).

Models the delegate skill's ``task_type`` auto-classification as DATA: if the
classified target field is still unset after arg parsing, the source field's
text is scanned (case-insensitive) for the first matching keyword group and
the corresponding value is assigned.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelSkillClassifier"]


class ModelSkillClassifier(BaseModel):
    """Keyword → payload-value classification for an unset argument."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_field: str = Field(
        ...,
        min_length=1,
        description="Payload field this classifier assigns when unset.",
    )
    source_field: str = Field(
        ...,
        min_length=1,
        description="Payload field whose text is scanned for keywords.",
    )
    rules: tuple[tuple[tuple[str, ...], str], ...] = Field(
        ...,
        description=(
            "Ordered ((keyword, ...), value) pairs. First group with any "
            "keyword present in the source text wins."
        ),
    )
    fallback: str = Field(
        ...,
        min_length=1,
        description="Value assigned when no keyword group matches.",
    )
