# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from enum import StrEnum


class EnumJobStatus(StrEnum):
    """The statuses an agent job record (``JobState``) can carry."""

    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SUPERSEDED = "superseded"
