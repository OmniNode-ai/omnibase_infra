# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where a board check can run.

``CI_EPHEMERAL`` runs against a stack a CI job starts; ``LAB_HARDWARE`` needs a
lab lane or lab hardware and runs in the lab proof; ``CLOUD`` and
``POST_RELEASE`` run after merge.

Ticket: OMN-19930
"""

from __future__ import annotations

from enum import StrEnum


class EnumBoardCheckSurfaceClass(StrEnum):
    """The surface class a board check declares in the node contract."""

    CI_EPHEMERAL = "ci_ephemeral"
    LAB_HARDWARE = "lab_hardware"
    CLOUD = "cloud"
    POST_RELEASE = "post_release"


__all__ = ["EnumBoardCheckSurfaceClass"]
