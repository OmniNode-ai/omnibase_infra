# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Result model for the Gmail Intent Poller Effect node.

Related Tickets:
    - OMN-2730: feat(omnibase_infra): add node_gmail_intent_poller_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType

__all__ = ["ModelGmailIntentPollerResult"]


class ModelGmailIntentPollerResult(BaseModel):
    """Result model for one Gmail intent poller run.

    Attributes:
        messages_processed: Total messages processed across all source labels.
        messages_archived: Total messages archived (moved to archive_label).
        hard_failed: True if any label-level list_messages call failed.
            Per-message failures (get_message / modify_labels) are captured
            in ``errors`` and do NOT set hard_failed.
        events_published: Set by the runtime after publishing pending_events.
            Invariant: ``events_published == len(pending_events)`` after
            the runtime publish step.
        errors: Per-message error strings (skip-and-continue failures).
        pending_events: Event payloads for the runtime to publish to
            ``onex.evt.omnibase_infra.gmail-intent-received.v1``.
            Invariant: ``events_published == len(pending_events)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    messages_processed: int = Field(
        default=0,
        ge=0,
        description="Total messages processed across all source labels.",
    )
    messages_archived: int = Field(
        default=0,
        ge=0,
        description="Total messages archived (moved to archive_label).",
    )
    hard_failed: bool = Field(
        default=False,
        description=(
            "True if any label-level list_messages call failed. "
            "Per-message failures are captured in errors, not hard_failed."
        ),
    )
    events_published: int = Field(
        default=0,
        ge=0,
        description=(
            "Set by the runtime after publishing pending_events. "
            "Invariant: events_published == len(pending_events)."
        ),
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Per-message error strings from skip-and-continue failures.",
    )
    pending_events: list[JsonType] = Field(
        default_factory=list,
        description=(
            "Event payloads for the runtime to publish. "
            "Invariant: events_published == len(pending_events)."
        ),
    )
