# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""What the off-registry omnimarket drift comparison concluded (OMN-17255)."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["EnumOffRegistryVerdict"]


class EnumOffRegistryVerdict(StrEnum):
    """The three outcomes of comparing installed packages against packaged pins.

    ``SKIPPED`` exits 0 like ``IN_SYNC`` and is deliberately a DIFFERENT value:
    "I checked and it matched" and "there was nothing to check" are exactly the
    two states the pre-OMN-17255 silent return collapsed into one.
    """

    IN_SYNC = "IN_SYNC"
    DRIFTED = "DRIFTED"
    SKIPPED = "SKIPPED"
