# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How a lane-added Bifrost backend shares work with the rungs it mirrors (OMN-19215).

A placement mirrors an added backend into a routing tier behind named rungs.
The routing authority (omnimarket) reads the mode: ``fallback`` offers the
added backend only when its rung is unroutable or already tried, ``spread``
also lets it take a share of the rung's first-choice traffic.
"""

from __future__ import annotations

from enum import Enum, unique

__all__ = ["EnumBifrostLanePlacementMode"]


@unique
class EnumBifrostLanePlacementMode(str, Enum):
    """Placement mode of a lane-added backend.

    Values:
        FALLBACK: Offered only when the rung it mirrors is unroutable or
            already tried. The default, and what an absent ``mode`` means.
        SPREAD: Also shares the rung's first-choice traffic; the routing
            authority picks one member of the group per request by a stable
            hash of the correlation id.
    """

    FALLBACK = "fallback"
    SPREAD = "spread"
