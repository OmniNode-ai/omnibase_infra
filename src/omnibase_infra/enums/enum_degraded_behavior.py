# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Degraded behavior enum for projection fallback strategy."""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumDegradedBehavior(str, Enum):
    """Fallback strategy to apply when a data source is unavailable or stale."""

    SERVE_STALE_WITH_WARNING = "serve_stale_with_warning"
    RETURN_EMPTY = "return_empty"
    FAIL_CLOSED = "fail_closed"


__all__ = ["EnumDegradedBehavior"]
