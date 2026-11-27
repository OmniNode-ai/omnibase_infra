# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which queued rebuild requests run, and which are superseded, operator-path first (OMN-26718).

On 2026-11-26 an operator-path rebuild request for dev (signed command c94ae2a7, 18:38Z) was
deferred to the 19:00Z recreate window, where the window admitted the automatic overdue job
instead and the operator request never ran. A request someone made on purpose must not wait
behind, or be replaced by, an automatic one.

This handler owns the static half of the deploy agent's coalescing decision (formerly
``coalesce._refusal_for``): which queued requests may fold into the head's group, and why the
first one that may not stops it. It adds the priority rule: on a lane whose queue is
superseded outright (OMN-26637), the first operator-path request in the head's lane runs and
the automatic requests queued before it are superseded, each with a reason that names it.
Ancestry stays with the agent: it needs a git clone, this does not.

Pure: ``handle(request) -> decision``, no clock, no I/O. The deploy agent loads this module
from its own checkout (``agent_entry``) and asks it once per poll.
"""

from __future__ import annotations

import re
from typing import Final

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_queued_request_entry import ModelQueuedRequestEntry
from omnibase_infra.models.model_queued_request_priority_decision import (
    ModelQueuedRequestPriorityDecision,
)
from omnibase_infra.models.model_queued_request_priority_request import (
    ModelQueuedRequestPriorityRequest,
)

#: A full 40-character lowercase commit sha; an abbreviated ref cannot be compared safely.
SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")


class HandlerQueuedRequestPriority:
    """Decide which queued requests fold and which runs (contract-bound, pure)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelQueuedRequestPriorityRequest
    ) -> ModelQueuedRequestPriorityDecision:
        entries = request.entries
        prefixes = request.operator_requester_prefixes
        head = entries[0]
        if request.supersede_queue:
            for index, entry in enumerate(entries):
                if entry.runtime_lane != head.runtime_lane:
                    break
                if entry.requested_by.startswith(prefixes):
                    return ModelQueuedRequestPriorityDecision(
                        foldable=index + 1,
                        refusal="operator_request" if index + 1 < len(entries) else "",
                        operator_path=True,
                        runner_index=index,
                        reasons=tuple(
                            f"superseded by {entry.correlation_id} at {entry.git_ref}, an "
                            "operator-path request, which runs ahead of an automatic "
                            "request for the same window"
                            for _ in entries[:index]
                        ),
                    )
        seen = {head.correlation_id}
        for position, candidate in enumerate(entries[1:], start=1):
            # OMN-19521: a redelivered copy of a request already in the group is not newer work.
            refusal = (
                "duplicate_command"
                if candidate.correlation_id in seen
                else self._static_refusal(request, head, candidate)
            )
            if refusal:
                return ModelQueuedRequestPriorityDecision(
                    foldable=position, refusal=refusal
                )
            seen.add(candidate.correlation_id)
        return ModelQueuedRequestPriorityDecision(foldable=len(entries))

    @staticmethod
    def _static_refusal(
        request: ModelQueuedRequestPriorityRequest,
        head: ModelQueuedRequestEntry,
        candidate: ModelQueuedRequestEntry,
    ) -> str:
        """The first static rule candidate breaks against head, or empty."""
        if candidate.runtime_lane != head.runtime_lane:
            return "different_lane"
        # OMN-26316: two overdue-build requests are one kind of build (the dev head, with a
        # rollback declared only to pass the lineage fence), so the newer replaces the older.
        overdue = (
            head.requested_by == candidate.requested_by == request.overdue_requester
        )
        if not overdue and (head.rollback_declared or candidate.rollback_declared):
            return "rollback_declared"
        if head.scope != candidate.scope or (not overdue and head.scope != "full"):
            return "not_full_scope"
        if candidate.services != head.services:
            return "services_differ"
        if candidate.build_source != head.build_source:
            return "build_source_differs"
        if head.pinned_image or candidate.pinned_image:
            return "pinned_image"
        if not SHA_RE.match(head.git_ref) or not SHA_RE.match(candidate.git_ref):
            return "ref_not_a_sha"
        return ""
