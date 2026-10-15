# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which authority supplied a rendered lane overlay (OMN-19214)."""

from __future__ import annotations

from enum import Enum, unique

__all__ = ["EnumBifrostLaneOverlaySource"]


@unique
class EnumBifrostLaneOverlaySource(str, Enum):
    """The source a render resolved its lane overlay from.

    ``STORE`` is the lane's row in the config store; ``FILE`` is the mounted
    overlay file. The render records the one it used so a fall back from an
    absent store row to the file is visible rather than silent.
    """

    STORE = "store"
    FILE = "file"
