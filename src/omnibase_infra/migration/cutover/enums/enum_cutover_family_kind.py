# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kinds of coherent database families participating in a cutover."""

from enum import StrEnum


class EnumCutoverFamilyKind(StrEnum):
    """Determines which continuity proof a family must carry."""

    PROJECTION = "projection"
    CONTROL_PLANE = "control_plane"


__all__ = ["EnumCutoverFamilyKind"]
