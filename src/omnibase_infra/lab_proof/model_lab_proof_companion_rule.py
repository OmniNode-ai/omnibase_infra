# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""What a bot-authored change-control companion is.

Ticket: OMN-19565
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLabProofCompanionRule(BaseModel):
    """A PR is a bot change-control companion only when all three hold.

    Its repository is ``repo``, its author is one of ``authors`` (a bot
    login, never a human), and every changed path matches ``path_globs``. A
    human-authored contract PR, or a bot PR that also touches code, is not one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(pattern=r"^OmniNode-ai/[A-Za-z0-9_.-]+$")
    authors: tuple[str, ...] = Field(min_length=1)
    path_globs: tuple[str, ...] = Field(min_length=1)


__all__ = ["ModelLabProofCompanionRule"]
