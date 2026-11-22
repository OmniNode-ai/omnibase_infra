# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from enum import StrEnum


class EnumPostFailureVerdict(StrEnum):
    CONVERGE = "converge"
    FROZEN = "frozen"
    JOB_ACTIVE = "job_active"
    NO_FAILED_JOB = "no_failed_job"
    SOURCE_UNKNOWN = "source_unknown"
    HEAD_UNREADABLE = "head_unreadable"
    UP_TO_DATE = "up_to_date"
    ALREADY_ATTEMPTED = "already_attempted"
    WINDOWS_UNREADABLE = "windows_unreadable"
    PROBE_WINDOW = "probe_window"
