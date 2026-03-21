# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer health incident lifecycle state enum.

Ticket: OMN-5511
"""

from __future__ import annotations

from enum import StrEnum


class EnumConsumerIncidentState(StrEnum):
    """Lifecycle state of a consumer health incident."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESTART_PENDING = "restart_pending"
    RESTART_SUCCEEDED = "restart_succeeded"
    RESTART_FAILED = "restart_failed"
    TICKETED = "ticketed"
    RESOLVED = "resolved"


__all__ = ["EnumConsumerIncidentState"]
