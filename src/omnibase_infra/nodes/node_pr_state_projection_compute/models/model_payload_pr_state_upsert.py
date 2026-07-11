# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelPayloadPrStateUpsert - intent payload for pr_state persistence.

The COMPUTE node emits a ModelIntent carrying this payload after parsing a
GitHub PR status event. The EFFECT node consumes the intent and upserts one
row into pr_state, keyed by (repo, pr_number).

SEAM (shared with the OMN-14374 structured-only read skill): this payload's
fields mirror the public.pr_state columns 1:1 -- see migration
091_pr_state.sql for the full target schema and the interim-producer caveat
(only triage_state/title/is_draft are populated by the current poller
producer; ci_status/review_decision/mergeable/merge_state_status/
merge_queue_state stay None until a richer producer lands).

SEAM MATCH (OMN-14394): ``is_draft`` is a non-nullable bool, matching
omnimarket's ``ModelOpenPrSummary.is_draft: bool`` (node_github_repo_gateway_
effect) field-for-field -- unlike the reserved columns above, the poller can
always determine draft status from GitHub's ``pr["draft"]`` today, so this
field defaults to ``False`` rather than ``None`` (see
092_pr_state_add_is_draft.sql).

CI_STATUS -> OVERALL MAPPING (documented, not yet implemented): when a future
producer populates ``ci_status``, the OMN-14374 reader's normalized
``_OverallState`` enum (``"green" | "red" | "pending"``) is the target shape.
The intended transform, so a future cutover has a specified mapping instead
of an implicit assumption:
    - GitHub check-run conclusion ``success`` -> ``green``
    - ``failure`` | ``cancelled`` | ``timed_out`` | ``action_required`` ->
      ``red``
    - ``neutral`` | ``skipped`` | ``stale`` | in-progress/queued/``None`` ->
      ``pending``
This is a raw-string passthrough today (``ci_status: str | None``); the
mapping above is the reader-side transform a consumer must apply, not
something this payload performs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadPrStateUpsert(BaseModel):
    """Payload for a single pr_state upsert.

    Attributes:
        intent_type: Discriminator literal, always "pr_state.upsert".
        repo: Repository identifier in "{owner}/{name}" format.
        pr_number: Pull request number.
        triage_state: Current triage classification (from the poller).
        title: Pull request title.
        is_draft: GitHub PR draft status. Always populated by the poller
            (unlike the reserved columns below) -- matches
            ModelOpenPrSummary.is_draft (omnimarket reader) 1:1.
        ci_status: Overall CI conclusion, when known. None if not yet
            populated by any producer (see module docstring).
        review_decision: GitHub reviewDecision (APPROVED, CHANGES_REQUESTED,
            etc.), when known.
        mergeable: GitHub mergeable state, when known.
        merge_state_status: GitHub mergeStateStatus, when known.
        merge_queue_state: Merge-queue entry state, when known.
        base_ref: Base branch name, when known.
        head_ref: Head branch name, when known.
        source: Producer provenance -- "poller" or "webhook".
        correlation_id: Distributed-trace correlation ID, when extracted.
        as_of: Producer-supplied event time (when GitHub reported this
            state) -- the freshness stamp readers check for staleness.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["pr_state.upsert"] = Field(
        default="pr_state.upsert",
        description="Discriminator literal for intent routing.",
    )

    repo: str = Field(
        ...,
        min_length=1,
        description="Repository identifier in '{owner}/{name}' format.",
    )
    pr_number: int = Field(
        ...,
        ge=1,
        description="Pull request number.",
    )
    triage_state: str = Field(
        ...,
        min_length=1,
        description="Current triage classification of the pull request.",
    )
    title: str = Field(
        default="",
        description="Pull request title.",
    )
    is_draft: bool = Field(
        default=False,
        description="GitHub PR draft status, mirrored from pr['draft'].",
    )
    ci_status: str | None = Field(
        default=None,
        description="Overall CI conclusion, when known.",
    )
    review_decision: str | None = Field(
        default=None,
        description="GitHub reviewDecision, when known.",
    )
    mergeable: str | None = Field(
        default=None,
        description="GitHub mergeable state, when known.",
    )
    merge_state_status: str | None = Field(
        default=None,
        description="GitHub mergeStateStatus, when known.",
    )
    merge_queue_state: str | None = Field(
        default=None,
        description="Merge-queue entry state, when known.",
    )
    base_ref: str | None = Field(
        default=None,
        description="Base branch name, when known.",
    )
    head_ref: str | None = Field(
        default=None,
        description="Head branch name, when known.",
    )
    source: str = Field(
        default="poller",
        description="Producer provenance -- 'poller' or 'webhook'.",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Distributed-trace correlation ID, when extracted.",
    )
    as_of: datetime = Field(
        ...,
        description="Producer-supplied event time -- the freshness stamp.",
    )


__all__ = ["ModelPayloadPrStateUpsert"]
