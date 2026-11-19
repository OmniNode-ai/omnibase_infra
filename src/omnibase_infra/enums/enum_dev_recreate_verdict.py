# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The verdict and the reason of the dev recreate window decision (OMN-25912)."""

from enum import StrEnum


class EnumDevRecreateVerdict(StrEnum):
    """Whether the dev runtime may be recreated now."""

    RECREATE = "recreate"
    HOLD = "hold"


class EnumDevRecreateReason(StrEnum):
    """Why the verdict holds; carried on the announcement the bus reads."""

    WINDOW_OPEN = "window_open"
    PINNED_FIX = "pinned_fix"
    WINDOW_SPENT = "window_spent"
    WINDOW_MISSED = "window_missed"
