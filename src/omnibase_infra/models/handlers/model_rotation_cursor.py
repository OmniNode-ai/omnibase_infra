# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rotating start position a handler keeps across runs outside itself (OMN-21536)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRotationCursor(BaseModel):
    """Where the next run of a rotated sequence begins.

    Each call to ``next_start`` returns the position this run starts at and moves
    the cursor one step on, so the handler that asks holds no position itself.
    """

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    position: int = Field(default=0, ge=0, description="Start of the next run.")

    def next_start(self, size: int) -> int:
        """Return this run's start in a sequence of ``size`` items, then advance."""
        start = self.position % size
        self.position = (start + 1) % size
        return start


__all__: list[str] = ["ModelRotationCursor"]
