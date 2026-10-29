# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The landing controller's lease rows of the lease inventory (OMN-21426)."""

from __future__ import annotations

from enum import StrEnum


class EnumLandingLeaseRow(StrEnum):
    """Inventory rows A1, A2, A3 and D5, migrated to the generic lease in slice 3.

    A8 (the cause claim ownership guard) has no row of its own: acquiring the
    ``landing-cause`` lease is the guard.
    """

    A1_FIX_WORKER = "A1"
    A2_CAUSE = "A2"
    A3_LAB_PROVER = "A3"
    D5_CI_RED_OWNER = "D5"


__all__ = ["EnumLandingLeaseRow"]
