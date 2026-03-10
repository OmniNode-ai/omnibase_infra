# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Domain model for GitHub PR webhook events consumed by the change detector."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPRWebhookEvent(BaseModel):
    """Represents a GitHub pull_request webhook payload (slim projection).

    Published to ``onex.evt.github.pr-webhook.v1`` by the GitHub Action
    webhook publisher script and consumed by HandlerPRWebhookIngestion.

    Only the fields required for change detection are captured. The full
    GitHub payload is intentionally not mirrored here.

    Related Tickets:
        - OMN-3940: Task 5 — Change Detector EFFECT Node
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["opened", "synchronize", "closed", "reopened", "edited"]
    repo: str = Field(
        description="Repository full name, e.g. 'OmniNode-ai/omnibase_infra'"
    )
    pr_number: int = Field(ge=1, description="Pull request number")
    head_ref: str = Field(description="Source branch name")
    head_sha: str = Field(description="Head commit SHA")
    changed_files: list[str] = Field(
        default_factory=list,
        description="File paths changed in this PR (populated from GitHub API)",
    )
    ticket_ids: list[str] = Field(
        default_factory=list,
        description="Linear ticket IDs extracted from PR title/body (e.g. ['OMN-1234'])",
    )
    actor: str | None = Field(
        default=None,
        description="GitHub login of the PR author or actor triggering the event",
    )
    merged: bool = Field(
        default=False,
        description="True when action=='closed' and the PR was merged",
    )


__all__ = ["ModelPRWebhookEvent"]
