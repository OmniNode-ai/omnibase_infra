# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Alert Severity Enumeration.

Defines severity levels for Slack alerts with corresponding
visual indicators for Block Kit formatting.
"""

from __future__ import annotations

from enum import Enum


class EnumAlertSeverity(str, Enum):
    """Alert severity levels for Slack notifications.

    Maps to visual indicators in Slack Block Kit messages:
        - CRITICAL: Red circle emoji (immediate attention required)
        - ERROR: Red circle emoji (error occurred)
        - WARNING: Yellow circle emoji (potential issue)
        - INFO: Blue circle emoji (informational)

    Attributes:
        CRITICAL: System-critical issue requiring immediate attention
        ERROR: Error condition that needs investigation
        WARNING: Warning condition that may need attention
        INFO: Informational message
    """

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


__all__ = ["EnumAlertSeverity"]
