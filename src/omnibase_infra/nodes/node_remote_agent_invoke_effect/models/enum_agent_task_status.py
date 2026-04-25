# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Agent task lifecycle status enum for the remote-agent invoke effect node."""

from __future__ import annotations

from enum import Enum


class EnumAgentTaskStatus(str, Enum):
    """Status values for remote agent task lifecycle events."""

    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


__all__ = ["EnumAgentTaskStatus"]
