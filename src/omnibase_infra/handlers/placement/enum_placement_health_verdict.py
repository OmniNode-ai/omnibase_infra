# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verdict of the per-node serving-count health read (OMN-21421)."""

from __future__ import annotations

from enum import Enum


class EnumPlacementHealthVerdict(str, Enum):
    """OK when every node is served by enough runtimes, FAIL otherwise."""

    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


__all__ = ["EnumPlacementHealthVerdict"]
