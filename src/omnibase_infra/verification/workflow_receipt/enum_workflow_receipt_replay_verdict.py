# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``EnumWorkflowReceiptReplayVerdict`` (OMN-15095)."""

from __future__ import annotations

from enum import Enum


class EnumWorkflowReceiptReplayVerdict(str, Enum):
    """PASS/FAIL verdict for one replay-then-diff run."""

    PASS = "PASS"
    FAIL = "FAIL"


__all__ = ["EnumWorkflowReceiptReplayVerdict"]
