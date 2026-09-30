# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stable subject grains for board probe result events (OMN-19937)."""

from __future__ import annotations

from enum import StrEnum


class EnumBoardSubjectKind(StrEnum):
    """The grain whose state one board probe result describes."""

    PR = "pr"
    MERGE_GROUP = "merge_group"
    BRANCH_HEAD = "branch_head"
    LANE = "lane"
    CHAIN = "chain"


__all__ = ["EnumBoardSubjectKind"]
