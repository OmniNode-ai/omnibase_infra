# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost projection snapshot kinds held by the cost API snapshot cache."""

from __future__ import annotations

from enum import Enum


class EnumCostSnapshotKind(str, Enum):
    """The cost projection snapshots the cost API reads before it queries the database.

    The cache is keyed by kind and window. A kind names the projection a snapshot
    came from; it is not a Kafka topic, so no topic string is declared here.
    """

    SUMMARY = "summary"
    BY_REPO = "by_repo"
    TOKEN_USAGE = "token_usage"


__all__ = ["EnumCostSnapshotKind"]
