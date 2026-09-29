# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The three outcomes of one board check execution.

``INDETERMINATE`` means the check could not decide. Every verdict path grades it
as a failure; there is no neutral or skipped outcome for a planned check.

Ticket: OMN-19930
"""

from __future__ import annotations

from enum import StrEnum


class EnumBoardProbeOutcome(StrEnum):
    """What one board check concluded about its subject."""

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


__all__ = ["EnumBoardProbeOutcome"]
