# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why the dev recreate window decided as it did (OMN-25912)."""

from enum import StrEnum


class EnumDevRecreateReason(StrEnum):
    """Why the verdict holds; carried on the announcement the bus reads."""

    WINDOW_OPEN = "window_open"
    PINNED_FIX = "pinned_fix"
    WINDOW_SPENT = "window_spent"
    WINDOW_MISSED = "window_missed"
