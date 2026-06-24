# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Disk watermark severity enum.

Ticket: OMN-13009
"""

from __future__ import annotations

from enum import Enum


class EnumDiskSeverity(str, Enum):
    """Severity classification for a disk watermark probe.

    Thresholds (usage_pct = used / total):
        CLEAN  — usage < 85%
        WARN   — 85% <= usage < 95%  → Slack alert
        P0     — usage >= 95%        → Urgent Linear ticket + Slack P0 alert

    The ``min_free_bytes`` cross-check can independently escalate a CLEAN or
    WARN probe to P0 when remaining free space is below redpanda's
    ``storage_min_free_bytes`` floor (default 10 MiB = 10485760 bytes).
    """

    CLEAN = "CLEAN"
    WARN = "WARN"
    P0 = "P0"


__all__ = ["EnumDiskSeverity"]
