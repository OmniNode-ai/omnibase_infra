# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Centralized Kafka topic constants for the emit daemon.

All topic strings used by the emit daemon and notification infrastructure
are defined here as module-level constants. This avoids duplication and
provides a single place to update topic names.
"""

TOPIC_PHASE_METRICS = "onex.evt.omniclaude.phase-metrics.v1"
TOPIC_NOTIFICATION_BLOCKED = "onex.evt.omniclaude.notification-blocked.v1"
TOPIC_NOTIFICATION_COMPLETED = "onex.evt.omniclaude.notification-completed.v1"

__all__: list[str] = [
    "TOPIC_PHASE_METRICS",
    "TOPIC_NOTIFICATION_BLOCKED",
    "TOPIC_NOTIFICATION_COMPLETED",
]
