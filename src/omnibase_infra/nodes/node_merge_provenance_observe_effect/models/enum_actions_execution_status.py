# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""GitHub Actions run and job status values (OMN-19927)."""

from __future__ import annotations

from enum import StrEnum


class EnumActionsExecutionStatus(StrEnum):
    """The ``status`` field of an Actions workflow run or job.

    A value outside this set fails model validation, which the observe effect
    reports as ``read_ok=False``: an unknown status is never guessed.
    """

    REQUESTED = "requested"
    QUEUED = "queued"
    PENDING = "pending"
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


__all__: list[str] = ["EnumActionsExecutionStatus"]
