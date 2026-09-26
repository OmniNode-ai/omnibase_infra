# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Retry a step until it succeeds or a deadline passes.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLabProofRetry(BaseModel):
    """Poll: re-run every ``interval_seconds`` until success or the deadline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    interval_seconds: int = Field(ge=1)
    deadline_seconds: int = Field(ge=1)


__all__ = ["ModelLabProofRetry"]
