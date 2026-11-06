# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to verify one fixed-in disposition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelFixVerifyRequest(BaseModel):
    """The commit a disposition names and the lines the thread cited."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha: str
    path: str
    original_commit: str = ""
    start: int | None = None
    end: int | None = None
    head: str
    base: str


__all__: list[str] = ["ModelFixVerifyRequest"]
