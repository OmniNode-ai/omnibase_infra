# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelPrStateUpsertResult - outcome of a single pr_state upsert.

pr_state is a latest-known-state projection (ON CONFLICT (repo, pr_number) DO
UPDATE), so `was_insert` distinguishes a first-seen PR from a refresh of an
existing row -- in contrast to ModelBuildLoopAppendResult, which has no such
flag because build_loop_runs is append-only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPrStateUpsertResult(BaseModel):
    """Outcome of one pr_state upsert."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the upsert completed without error.",
    )
    repo: str = Field(
        ...,
        min_length=1,
        description="Repository identifier persisted to the row.",
    )
    pr_number: int = Field(
        ...,
        ge=1,
        description="Pull request number persisted to the row.",
    )
    was_insert: bool = Field(
        ...,
        description="True if this PR had no prior row (first-seen); False if "
        "an existing row was refreshed.",
    )


__all__ = ["ModelPrStateUpsertResult"]
