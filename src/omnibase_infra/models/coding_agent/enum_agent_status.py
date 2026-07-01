# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminal status of a coding-agent invocation (OMN-13247)."""

from __future__ import annotations

from enum import Enum


class EnumAgentStatus(str, Enum):
    """System-derived terminal status of one coding-agent invocation.

    This is authoritative (set from the subprocess exit + git-derived diff), not
    agent-reported. ``REJECTED`` means workspace validation failed and no
    subprocess ever ran.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
