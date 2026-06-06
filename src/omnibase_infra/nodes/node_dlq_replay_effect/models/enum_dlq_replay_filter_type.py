# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Filter-type enum for DLQ replay eligibility selection (OMN-12619)."""

from __future__ import annotations

from enum import Enum


class EnumDlqReplayFilterType(str, Enum):
    """Selective replay filter strategy.

    Determines which type-specific predicate ``should_replay()`` applies on top
    of the always-on max-retry, non-retryable, and time-range checks.
    """

    ALL = "all"
    BY_TOPIC = "by_topic"
    BY_ERROR_TYPE = "by_error_type"
    BY_TIME_RANGE = "by_time_range"
    BY_CORRELATION_ID = "by_correlation_id"


__all__ = ["EnumDlqReplayFilterType"]
