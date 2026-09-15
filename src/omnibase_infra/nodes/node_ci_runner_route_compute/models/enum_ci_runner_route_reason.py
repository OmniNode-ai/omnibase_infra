# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Why the router decided what it did.

Ticket: OMN-18412
"""

from __future__ import annotations

from enum import StrEnum


class EnumCIRunnerRouteReason(StrEnum):
    """Why the router decided what it did."""

    # --- placement decided before any capacity signal is read ---------------
    POLICY_ALLOWLIST = "policy_allowlist"
    FORK_ISOLATION = "fork_isolation"
    SEAM_CEILING_HOSTED = "seam_ceiling_hosted"
    FORCED_FLEET = "forced_fleet"
    FORCED_HOSTED = "forced_hosted"
    # --- capacity ------------------------------------------------------------
    PROBE_ERROR = "probe_error"
    FLEET_SATURATED = "fleet_saturated"
    FLEET_DEGRADED = "fleet_degraded"
    LAB_UNKNOWN = "lab_unknown"
    LAB_SATURATED = "lab_saturated"
    CAPACITY_AVAILABLE = "capacity_available"
    # --- guards --------------------------------------------------------------
    NEVER_WIDEN_VIOLATION = "never_widen_violation"
    PRIVATE_REPO_NO_HOSTED_DOWNGRADE = "private_repo_no_hosted_downgrade"
    VISIBILITY_UNKNOWN_NO_HOSTED_DOWNGRADE = "visibility_unknown_no_hosted_downgrade"
    PRIVATE_REPO_HOSTED_FORBIDDEN = "private_repo_hosted_forbidden"
    PRIVATE_REPO_CONVERSION_WOULD_WIDEN = "private_repo_conversion_would_widen"


__all__ = ["EnumCIRunnerRouteReason"]
