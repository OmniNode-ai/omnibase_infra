# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import StrEnum


class EnumProjectTrackerIssueStatusType(StrEnum):
    """Linear workflow state type values."""

    BACKLOG = "backlog"
    UNSTARTED = "unstarted"
    STARTED = "started"
    COMPLETED = "completed"
    CANCELED = "canceled"


__all__: list[str] = [
    "EnumProjectTrackerIssueStatusType",
]
