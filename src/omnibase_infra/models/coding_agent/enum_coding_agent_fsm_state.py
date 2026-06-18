# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The coding-agent workflow FSM states (OMN-13247, plan §5.2).

States owned by ``node_coding_agent_fsm_reducer``:

    IDLE -> VALIDATING -> INVOKING -> CAPTURING -> COMPLETED | FAILED | REJECTED
"""

from __future__ import annotations

from enum import Enum


class EnumCodingAgentFsmState(str, Enum):
    """Deterministic FSM states for one coding-agent invocation."""

    IDLE = "idle"
    VALIDATING = "validating"
    INVOKING = "invoking"
    CAPTURING = "capturing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


TERMINAL_STATES: frozenset[EnumCodingAgentFsmState] = frozenset(
    {
        EnumCodingAgentFsmState.COMPLETED,
        EnumCodingAgentFsmState.FAILED,
        EnumCodingAgentFsmState.REJECTED,
    }
)
