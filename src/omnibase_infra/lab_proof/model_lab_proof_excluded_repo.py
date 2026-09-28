# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""A repository deliberately left out of the proof registry, with its reason.

Ticket: OMN-19565
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLabProofExcludedRepo(BaseModel):
    """A repository that takes no pull requests, named so its absence is visible."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(pattern=r"^OmniNode-ai/[A-Za-z0-9_.-]+$")
    reason: str = Field(min_length=1)


__all__ = ["ModelLabProofExcludedRepo"]
