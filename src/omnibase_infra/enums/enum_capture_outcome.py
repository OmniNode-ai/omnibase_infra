# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Capture outcome enum for corpus capture service.

.. versionadded:: 0.5.0
    Added for CorpusCapture (OMN-1203)
"""

from enum import Enum


class EnumCaptureOutcome(str, Enum):
    """Outcome of a capture attempt."""

    CAPTURED = "captured"
    """Execution was successfully captured."""

    SKIPPED_HANDLER_FILTER = "skipped_handler_filter"
    """Skipped due to handler filter (not in whitelist)."""

    SKIPPED_TIME_WINDOW = "skipped_time_window"
    """Skipped due to time window filter."""

    SKIPPED_SAMPLE_RATE = "skipped_sample_rate"
    """Skipped due to sampling (random selection)."""

    SKIPPED_DUPLICATE = "skipped_duplicate"
    """Skipped due to deduplication."""

    SKIPPED_CORPUS_FULL = "skipped_corpus_full"
    """Skipped because corpus is at max_executions."""

    SKIPPED_NOT_CAPTURING = "skipped_not_capturing"
    """Skipped because capture is not active."""

    SKIPPED_TIMEOUT = "skipped_timeout"
    """Skipped because capture operation timed out."""

    FAILED = "failed"
    """Capture failed due to an error."""
