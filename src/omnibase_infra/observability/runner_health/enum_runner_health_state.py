# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner health state enum."""

from __future__ import annotations

from enum import StrEnum


class EnumRunnerHealthState(StrEnum):
    """Computed health state for a single runner."""

    HEALTHY = "healthy"
    GITHUB_OFFLINE = "github_offline"
    DOCKER_UNHEALTHY = "docker_unhealthy"
    CRASH_LOOPING = "crash_looping"
    STALE_REGISTRATION = "stale_registration"
    MISSING = "missing"
