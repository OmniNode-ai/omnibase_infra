# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Frozen Pydantic model for GitHub PR merge events.

Published to ``onex.evt.github.pr-merged.v1`` by the GitHub Actions workflow
``pr-merged-event.yml`` and the companion publisher script
``scripts/publish_pr_merged_event.py``.

Only the fields required by downstream consumers are captured. The full
GitHub webhook payload is intentionally not mirrored here.

Related Tickets:
    - OMN-6726: GitHub merge event producer (webhook -> Kafka)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelPRMergedEvent(BaseModel):
    """Represents a GitHub pull_request merged event (slim projection).

    Published to ``onex.evt.github.pr-merged.v1`` by the GitHub Actions
    workflow when a PR is closed with ``merged == true``.

    Related Tickets:
        - OMN-6726: GitHub merge event producer
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(
        description="Repository full name, e.g. 'OmniNode-ai/omnibase_infra'",
    )
    pr_number: int = Field(ge=1, description="Pull request number")
    base_ref: str = Field(description="Target branch name (e.g. 'main')")
    head_ref: str = Field(description="Source branch name")
    merge_sha: str = Field(description="Merge commit SHA")
    author: str = Field(description="GitHub login of the PR author")
    changed_files: list[str] = Field(
        default_factory=list,
        description="File paths changed in this PR (populated from GitHub API)",
    )
    ticket_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Linear ticket IDs extracted from PR title/body/branch (e.g. ['OMN-1234'])"
        ),
    )
    title: str = Field(
        default="",
        description="PR title",
    )
    merged_at: datetime | None = Field(
        default=None,
        description="Timestamp when the PR was merged (ISO 8601)",
    )


__all__ = ["ModelPRMergedEvent"]
