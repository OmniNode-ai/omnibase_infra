# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of verifying one fixed-in disposition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelFixVerifyResult(BaseModel):
    """Whether the commit is a fix of this PR for the cited lines, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    reason: str


__all__: list[str] = ["ModelFixVerifyResult"]
