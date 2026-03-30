# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session registry status enum.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 3).
"""

from __future__ import annotations

from enum import StrEnum


class EnumSessionRegistryStatus(StrEnum):
    """Status of a session registry entry.

    Values:
        ACTIVE: Task is actively being worked on.
        COMPLETED: Task has been completed.
        STALLED: Task has had no activity for extended period.
    """

    ACTIVE = "active"
    COMPLETED = "completed"
    STALLED = "stalled"
