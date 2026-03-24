# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime error severity level enum.

Ticket: OMN-5513
"""

from __future__ import annotations

from enum import StrEnum


class EnumRuntimeErrorSeverity(StrEnum):
    """Severity level for runtime error events."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


__all__ = ["EnumRuntimeErrorSeverity"]
