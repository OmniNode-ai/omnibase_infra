# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build loop phase enumeration for the autonomous build loop FSM.

Related:
    - OMN-7311: ModelBuildLoopState foundation models
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from enum import Enum


class EnumBuildLoopPhase(str, Enum):
    """FSM phases for the autonomous build loop.

    Phase Transitions:
        IDLE -> CLOSING_OUT: Build loop started, close out pending work first
        CLOSING_OUT -> DEPLOYING: Close-out complete, trigger rebuild if PRs merged
        DEPLOYING -> VERIFYING: Deploy triggered (or skipped), verify system health
        VERIFYING -> FILLING: Verification passed, fill sprint backlog
        FILLING -> CLASSIFYING: Backlog filled, classify tickets
        CLASSIFYING -> BUILDING: Tickets classified, dispatch builds
        BUILDING -> COMPLETE: All builds dispatched
        COMPLETE -> IDLE: Cycle finished, ready for next
        Any -> FAILED: Unrecoverable error or circuit breaker tripped
    """

    IDLE = "idle"
    CLOSING_OUT = "closing_out"
    DEPLOYING = "deploying"
    VERIFYING = "verifying"
    FILLING = "filling"
    CLASSIFYING = "classifying"
    BUILDING = "building"
    COMPLETE = "complete"
    FAILED = "failed"


__all__: list[str] = ["EnumBuildLoopPhase"]
