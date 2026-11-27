# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The queued requests waiting for one window, oldest first (OMN-26718)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_queued_request_entry import ModelQueuedRequestEntry


class ModelQueuedRequestPriorityRequest(BaseModel):
    """Queue order, head first; the head is the request the agent is about to accept."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[ModelQueuedRequestEntry, ...] = Field(min_length=1)
    #: The head's lane replaces every older queued request with the newest (OMN-26637).
    supersede_queue: bool = False
    #: ``requested_by`` prefixes of an operator-path request: one someone made on purpose.
    operator_requester_prefixes: tuple[str, ...] = ("operator",)
    #: The requester of the overdue dev merge build check (OMN-26316): its requests are one
    #: kind of build, so the newer replaces the older at one scope.
    overdue_requester: str = "agent/overdue-dev-merge-build"
