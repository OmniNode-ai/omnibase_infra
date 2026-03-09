# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result model for PR comment posting operations.

Tracking:
    - OMN-3944: Task 7 — Reconciliation ORCHESTRATOR Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelPRCommentResult(BaseModel):
    """Result of a PR comment post or update operation.

    Returned by HandlerPlanToPRComment after attempting to post or update
    a GitHub PR comment with the artifact update plan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    posted: bool
    """Whether the comment was successfully created or updated."""

    comment_id: int | None = None
    """GitHub comment ID if the comment was posted or updated."""

    error: str | None = None
    """Error message if the post failed (HTTP error or network error)."""

    skipped: bool = False
    """True if the trigger type is not a PR trigger and posting was skipped."""


__all__: list[str] = ["ModelPRCommentResult"]
