# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-signal outcome for composite runner readiness (OMN-15255).

Tri-state because two-state loses the distinction that matters: a probe that
returned "bad" and a probe that never returned are different facts, and
collapsing them is exactly the fail-open bug OMN-14228 Slice A fixed at the
source level (a failed docker probe reading as ``docker_restart_count=0``).
"""

from __future__ import annotations

from enum import StrEnum


class EnumReadinessSignalOutcome(StrEnum):
    """Outcome of one readiness signal for one runner."""

    PASS = "pass"
    """The signal was probed and the runner satisfies it."""

    FAIL = "fail"
    """The signal was probed and the runner does NOT satisfy it."""

    UNKNOWN = "unknown"
    """The signal could not be determined (fact absent or its source failed)."""


__all__ = ["EnumReadinessSignalOutcome"]
