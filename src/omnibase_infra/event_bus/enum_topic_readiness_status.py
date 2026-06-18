# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic readiness status enum for the per-contract boot interleave (OMN-13237).

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from enum import Enum


class EnumTopicReadinessStatus(str, Enum):
    """Outcome of a single contract's topic-set readiness confirm.

    Values:
        READY: All of the contract's topics converged on broker metadata.
        NOT_READY: At least one topic did not converge within the budget.
        UNAVAILABLE: The broker/admin client could not be reached at all.
        SKIPPED: Readiness confirm was skipped (no provisioner / no topics).
    """

    READY = "ready"
    NOT_READY = "not_ready"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


__all__: list[str] = ["EnumTopicReadinessStatus"]
