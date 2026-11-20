# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The mount types ``docker compose config --format json`` renders for a service volume."""

from __future__ import annotations

from enum import StrEnum


class EnumRunnerMountType(StrEnum):
    """Long-form volume ``type`` values of a rendered Compose service."""

    BIND = "bind"
    VOLUME = "volume"
    TMPFS = "tmpfs"
    NPIPE = "npipe"
    IMAGE = "image"
    CLUSTER = "cluster"
