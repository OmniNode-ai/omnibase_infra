# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DLQ Replay Status Enum.

This module provides the status enum for DLQ replay operations.

Related:
    - scripts/dlq_replay.py - CLI tool that uses this enum
    - OMN-1032 - PostgreSQL tracking integration ticket
"""

from __future__ import annotations

from enum import Enum


class EnumReplayStatus(str, Enum):
    """Status of a DLQ replay operation.

    This enum tracks the lifecycle of a replay attempt:
    - PENDING: Replay has been initiated but not yet completed
    - COMPLETED: Message was successfully replayed to target topic
    - FAILED: Replay attempt failed (will be recorded with error_message)
    - SKIPPED: Message was intentionally not replayed (e.g., non-retryable error)

    Usage:
        This enum is the canonical definition for replay status tracking.
        It is imported by scripts/dlq_replay.py for CLI usage.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


__all__: list[str] = ["EnumReplayStatus"]
