# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Slack Alert Input Payload Model.

Provides the input payload model for Slack alert operations.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.handlers.models.enum_alert_severity import EnumAlertSeverity


class ModelSlackAlert(BaseModel):
    """Input payload for Slack alert operations.

    This model defines the structure of alert payloads sent to
    the HandlerSlackWebhook. The handler transforms this into
    Slack Block Kit formatted messages.

    Attributes:
        severity: Alert severity level for visual formatting
        message: Main alert message (required)
        title: Optional alert title (defaults to severity-based title)
        details: Optional key-value details to include in the alert
        channel: Optional channel override (webhook default if not specified)
        correlation_id: UUID for distributed tracing

    Example:
        >>> alert = ModelSlackAlert(
        ...     severity=EnumAlertSeverity.WARNING,
        ...     message="High memory usage detected",
        ...     title="Resource Alert",
        ...     details={"node": "registry-effect", "memory_pct": "85"},
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    severity: EnumAlertSeverity = Field(
        default=EnumAlertSeverity.INFO,
        description="Alert severity level for visual formatting",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="Main alert message content",
    )
    title: str | None = Field(
        default=None,
        max_length=150,
        description="Optional alert title (defaults to severity-based title)",
    )
    details: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Additional key-value details to include in the alert",
    )
    channel: str | None = Field(
        default=None,
        description="Optional channel override (uses webhook default if not specified)",
    )
    thread_ts: str | None = Field(
        default=None,
        description="Reply in this Slack thread (ts value). If None, creates new top-level message.",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for distributed tracing",
    )


__all__ = ["ModelSlackAlert"]
