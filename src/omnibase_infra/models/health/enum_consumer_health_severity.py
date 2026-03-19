# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer health severity level enum.

Ticket: OMN-5511
"""

from __future__ import annotations

from enum import StrEnum


class EnumConsumerHealthSeverity(StrEnum):
    """Severity level for consumer health events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


__all__ = ["EnumConsumerHealthSeverity"]
