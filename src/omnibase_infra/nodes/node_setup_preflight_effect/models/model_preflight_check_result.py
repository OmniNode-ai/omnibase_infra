# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Result model for a single preflight check.

Ticket: OMN-3491
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPreflightCheckResult(BaseModel):
    """Result of a single preflight check.

    Frozen and immutable — safe for tuple collection in output models.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    check_key: str = Field(
        ...,
        description="Key identifying the preflight check performed (e.g. 'postgres_reachable').",
    )
    passed: bool = Field(
        ...,
        description="Whether the check passed.",
    )
    message: str = Field(
        ...,
        description="Human-readable result summary.",
    )
    detail: str | None = Field(
        default=None,
        description="Optional extended detail or error context.",
    )


__all__: list[str] = ["ModelPreflightCheckResult"]
