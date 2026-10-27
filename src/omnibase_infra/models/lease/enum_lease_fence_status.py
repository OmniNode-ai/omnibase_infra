# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The answer of ``check_fence`` for one fence (OMN-21424)."""

from __future__ import annotations

from enum import StrEnum


class EnumLeaseFenceStatus(StrEnum):
    """Whether a fence still names the live grant of its resource.

    ``OK``: the fence is the live grant's. ``STALE``: a newer grant holds the
    resource. ``EXPIRED``: no grant holds the resource. ``UNKNOWN``: the store
    could not be read, or the fence is above every live grant's.
    """

    OK = "ok"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


__all__ = ["EnumLeaseFenceStatus"]
