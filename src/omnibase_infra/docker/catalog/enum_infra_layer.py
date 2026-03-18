# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Constrained set of infrastructure layers for the service catalog."""

from __future__ import annotations

from enum import Enum


class EnumInfraLayer(str, Enum):
    """Constrained set of infrastructure layers."""

    INFRASTRUCTURE = "infrastructure"
    RUNTIME = "runtime"
    OBSERVABILITY = "observability"
