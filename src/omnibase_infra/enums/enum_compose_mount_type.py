# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker Compose long-syntax volume mount types."""

from enum import Enum


class EnumComposeMountType(str, Enum):
    """Mount ``type`` values accepted by Compose long-syntax volumes."""

    BIND = "bind"
    VOLUME = "volume"
    TMPFS = "tmpfs"
    NPIPE = "npipe"
    CLUSTER = "cluster"
    IMAGE = "image"
