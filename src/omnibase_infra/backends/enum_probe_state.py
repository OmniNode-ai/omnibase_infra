# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Probe state enum for onex.backends health probes."""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumProbeState(str, Enum):
    """4-state probe result for backend health."""

    DISCOVERED = "DISCOVERED"
    REACHABLE = "REACHABLE"
    HEALTHY = "HEALTHY"
    AUTHORITATIVE = "AUTHORITATIVE"
