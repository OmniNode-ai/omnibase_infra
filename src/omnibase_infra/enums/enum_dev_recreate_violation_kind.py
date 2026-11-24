# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which dev recreate rule a start broke (OMN-25912)."""

from enum import StrEnum


class EnumDevRecreateViolationKind(StrEnum):
    """The four cases the dev recreate reader flags."""

    TWO_STARTS_IN_WINDOW = "two_starts_in_window"
    OFF_WINDOW_START = "off_window_start"
    UNANNOUNCED_START = "unannounced_start"
    START_WITHOUT_JOB_RECORD = "start_without_job_record"
