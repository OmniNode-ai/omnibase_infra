# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The answer of ``check_fence`` for one fence (OMN-21424)."""

from __future__ import annotations

from enum import StrEnum


class EnumLeaseFenceStatus(StrEnum):
    """Whether a fence still names the live grant of its resource.

    ``OK``: the fence is the live grant's. ``STALE``: a grant with a higher
    fence holds the resource. ``EXPIRED``: no live grant has this fence or a
    higher one. ``UNKNOWN``: the store could not be read.
    """

    OK = "ok"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


__all__ = ["EnumLeaseFenceStatus"]
