# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR status event the GitHub PR poller publishes, one per open pull request.

Related Tickets:
    - OMN-2656: Phase 2 — Effect Nodes & CLIs (omnibase_infra)
    - OMN-20867: only these events reach onex.evt.github.pr-status.v1
"""

from __future__ import annotations

from pydantic import Field

from omnibase_core.models.events.model_github_pr_status_event import (
    ModelGitHubPRStatusEvent,
)
from omnibase_infra.topics import SUFFIX_GITHUB_PR_STATUS

__all__ = ["ModelGitHubPrStatusPollEvent"]


class ModelGitHubPrStatusPollEvent(ModelGitHubPRStatusEvent):
    """A ``ModelGitHubPRStatusEvent`` as the poller observes it.

    Adds the two fields the poller has always put on the wire: ``is_draft``
    (OMN-14394, read by the PR-state projection) and ``partition_key``.

    Attributes:
        event_type: The topic this event is published to.
        is_draft: GitHub's draft flag for the pull request.
        partition_key: ``{repo}:{pr_number}``.
    """

    event_type: str = Field(
        default=SUFFIX_GITHUB_PR_STATUS,
        description="The topic this event is published to.",
    )
    is_draft: bool = Field(
        default=False,
        description="GitHub's draft flag for the pull request.",
    )
    partition_key: str = Field(
        ...,
        min_length=3,
        description="'{repo}:{pr_number}', so one PR's events stay ordered.",
    )
