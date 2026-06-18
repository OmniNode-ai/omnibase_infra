# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Classified topic readiness failure reason (OMN-13237, §3.7).

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from enum import Enum


class EnumTopicReadinessFailureReason(str, Enum):
    """Classified reason a topic failed its readiness confirm (§3.7).

    Values:
        TOPIC_ABSENT: Broker metadata never returned the topic.
        PARTITION_MISMATCH: Partition count did not match the expected spec.
        NO_LEADER: At least one partition had no available leader.
        CONFIG_MISMATCH: A required topic config key was not visible/accepted.
        REPLICATION_MISMATCH: Reported replication factor did not match spec.
    """

    TOPIC_ABSENT = "topic_absent"
    PARTITION_MISMATCH = "partition_mismatch"
    NO_LEADER = "no_leader"
    CONFIG_MISMATCH = "config_mismatch"
    REPLICATION_MISMATCH = "replication_mismatch"


__all__: list[str] = ["EnumTopicReadinessFailureReason"]
