# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Result model for the Gmail Archive Cleanup Effect node.

Related Tickets:
    - OMN-2731: Add node_gmail_archive_cleanup_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType

__all__ = ["ModelGmailCleanupResult"]


class ModelGmailCleanupResult(BaseModel):
    """Result from one archive cleanup run.

    Attributes:
        purged_count: Total number of messages permanently deleted across
            all archive labels.
        label_counts: Per-label count of messages deleted. Keys are label
            names (not IDs); values are delete counts.
        hard_failed: True if any label's search_messages call raised an
            exception (the label was skipped entirely). Individual message
            delete failures do not set this flag.
        errors: Non-fatal error messages for individual delete failures.
            Also includes hard-failure messages for skipped labels.
        events_published: Number of summary events published by the
            node shell / runtime. Set by the runtime after publishing
            pending_events.
        pending_events: List of event payloads for the runtime to publish.
            The handler always returns at most one payload (the summary
            event) when purged_count > 0 or any errors exist.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    purged_count: int = Field(
        default=0,
        description="Total messages permanently deleted across all archive labels.",
    )
    label_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Per-label count of messages deleted (label name → count).",
    )
    hard_failed: bool = Field(
        default=False,
        description=(
            "True if any label's search_messages call raised an exception "
            "and the label was skipped entirely."
        ),
    )
    errors: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal error messages for individual delete failures and "
            "hard-failure messages for skipped labels."
        ),
    )
    events_published: int = Field(
        default=0,
        description=(
            "Number of summary events published by the runtime. Set by the "
            "runtime after publishing pending_events."
        ),
    )
    pending_events: list[JsonType] = Field(
        default_factory=list,
        description=(
            "Event payloads pending publication by the runtime/node shell. "
            "At most one payload per cleanup run."
        ),
    )
