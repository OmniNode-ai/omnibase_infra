# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Input configuration model for the Gmail Archive Cleanup Effect node.

Related Tickets:
    - OMN-2731: Add node_gmail_archive_cleanup_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelGmailCleanupConfig"]


class ModelGmailCleanupConfig(BaseModel):
    """Configuration for the Gmail archive cleanup operation.

    Attributes:
        archive_labels: Label names to purge from. The handler resolves
            these to Gmail label IDs via HandlerGmailApi.resolve_label_ids.
        retention_days: Age threshold in days. Messages older than this
            cutoff will be permanently deleted. Must be between 1 and 365.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    archive_labels: list[str] = Field(
        description="Label names to purge from.",
    )
    retention_days: int = Field(
        default=60,
        ge=1,
        le=365,
        description=(
            "Age threshold in days. Messages older than this cutoff are "
            "permanently deleted. Must be between 1 and 365."
        ),
    )
