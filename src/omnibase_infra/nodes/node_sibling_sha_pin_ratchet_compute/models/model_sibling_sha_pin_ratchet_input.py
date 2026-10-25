# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Explicit, immutable inputs collected by the sha-pin ratchet runtime."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelSiblingShaPinRatchetInput"]


class ModelSiblingShaPinRatchetInput(BaseModel):
    """Candidate lines of the change and of its base; the base is the baseline.

    Each entry is a repository-relative path and one line of that file that
    holds a 40-hex token. The runtime collects them with git grep at the base
    ref and in the change, so no baseline file exists to grow.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    base_ref: str = Field(min_length=1, description="The base ref, for messages.")
    base_lines: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Path and text of every base-ref line holding a 40-hex token.",
    )
    lines: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Path and text of every changed-tree line holding a 40-hex token.",
    )
