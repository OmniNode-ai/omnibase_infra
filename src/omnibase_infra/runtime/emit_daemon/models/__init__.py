# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Emit Daemon Event Models.

Pydantic models for notification events emitted by the
emit daemon and consumed by the notification consumer.

Related Tickets:
    - OMN-1831: Implement event-driven Slack notifications via runtime
"""

from omnibase_infra.runtime.emit_daemon.models.model_notification_blocked import (
    ModelNotificationBlocked,
)
from omnibase_infra.runtime.emit_daemon.models.model_notification_completed import (
    ModelNotificationCompleted,
)

__all__ = [
    "ModelNotificationBlocked",
    "ModelNotificationCompleted",
]
