# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""FSM transition trigger an event carries (OMN-13247, plan §5.2).

Event kinds map 1:1 to the FSM transition triggers:

    invoke-requested -> VALIDATING
    workspace-rejected -> REJECTED (no subprocess ever runs)
    workspace-ok -> INVOKING
    invoke-completed -> CAPTURING
    invoke-failed -> retry-or-FAILED per policy
    diff-captured -> COMPLETED
"""

from __future__ import annotations

from enum import Enum


class EnumCodingAgentEventKind(str, Enum):
    """The FSM transition trigger an event carries."""

    INVOKE_REQUESTED = "invoke_requested"
    WORKSPACE_REJECTED = "workspace_rejected"
    WORKSPACE_OK = "workspace_ok"
    INVOKE_COMPLETED = "invoke_completed"
    INVOKE_FAILED = "invoke_failed"
    DIFF_CAPTURED = "diff_captured"
