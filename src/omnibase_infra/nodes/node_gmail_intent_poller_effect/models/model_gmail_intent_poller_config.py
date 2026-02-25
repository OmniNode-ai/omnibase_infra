# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for the Gmail Intent Poller Effect node.

Related Tickets:
    - OMN-2730: feat(omnibase_infra): add node_gmail_intent_poller_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelGmailIntentPollerConfig"]


class ModelGmailIntentPollerConfig(BaseModel):
    """Configuration model for the Gmail intent poller effect node.

    Declares which Gmail labels to drain, how to archive processed messages,
    and the idempotency marker label applied before emitting events.

    Attributes:
        source_labels: Label names to drain (e.g. ["to-read"]).
        archive_label: Label name to apply after processing (archiving).
        processed_label: Idempotency marker label applied BEFORE emitting
            events. Messages with this label still in source during the
            recovery pass are archived without re-emitting.
        max_per_label: Maximum messages to process per label per run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    source_labels: list[str] = Field(
        ...,
        description="Label names to drain (e.g. ['to-read']).",
    )
    archive_label: str = Field(
        ...,
        description="Label name to apply after processing (archiving).",
    )
    processed_label: str = Field(
        ...,
        description=(
            "Idempotency marker label applied BEFORE emitting events. "
            "Messages with this label still in source = crashed mid-run; "
            "recovery pass archives without re-emitting."
        ),
    )
    max_per_label: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum messages to process per label per run.",
    )
