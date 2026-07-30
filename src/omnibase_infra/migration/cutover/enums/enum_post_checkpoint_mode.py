# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Allowed behavior after the first target-only authoritative write."""

from enum import StrEnum


class EnumPostCheckpointMode(StrEnum):
    """Per-family rollback posture after source authority becomes stale."""

    REVERSE_DELTA = "reverse_delta"
    FORWARD_FIX_ONLY = "forward_fix_only"


__all__ = ["EnumPostCheckpointMode"]
