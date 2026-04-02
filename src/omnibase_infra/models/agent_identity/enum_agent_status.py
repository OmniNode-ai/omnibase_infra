# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Agent lifecycle status."""

from enum import StrEnum


class EnumAgentStatus(StrEnum):
    """Status of a persistent agent entity."""

    IDLE = "idle"
    ACTIVE = "active"
    SUSPENDED = "suspended"
