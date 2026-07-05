# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-runner health state enum for the runner-fleet-maintain workflow (OMN-13942).

Distinct from ``omnibase_infra.observability.runner_health.enum_runner_health_state
.EnumRunnerHealthState`` (the legacy CLI collector's classification). That enum is
untouched -- ``runner-monitor.sh``/``collector_runner_health.py`` keep running
unmodified until this node is proven equivalent-or-better over a real incident
window.
"""

from __future__ import annotations

from enum import StrEnum


class EnumRunnerFleetHealthState(StrEnum):
    """Computed per-runner health state for the runner-fleet-maintain COMPUTE node."""

    HEALTHY = "healthy"
    OFFLINE_IDLE = "offline_idle"
    LISTENER_ZOMBIE = "listener_zombie"
    CRASH_LOOPING = "crash_looping"
    WEDGED = "wedged"
    SATURATED = "saturated"
    # NEW checks closing the OMN-13932 gap -- no equivalent exists in the
    # legacy bash/CLI surface.
    BUILDX_UNAVAILABLE = "buildx_unavailable"
    CODELOAD_THROTTLED = "codeload_throttled"


__all__ = ["EnumRunnerFleetHealthState"]
