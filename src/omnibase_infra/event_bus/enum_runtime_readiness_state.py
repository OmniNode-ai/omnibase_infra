# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregate runtime readiness tri-state (OMN-13237, §3.8).

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from enum import Enum


class EnumRuntimeReadinessState(str, Enum):
    """Aggregate runtime readiness tri-state (§3.8).

    Liveness is a separate signal and stays true regardless of this state.

    Values:
        READY: All required contracts attached.
        DEGRADED: attached_contracts < required_contracts (non-core gap).
        FAILED: A core control-plane contract could not attach.
    """

    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


__all__: list[str] = ["EnumRuntimeReadinessState"]
