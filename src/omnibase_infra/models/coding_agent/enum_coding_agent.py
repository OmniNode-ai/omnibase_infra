# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which coding-agent CLI a coding-agent command targets (OMN-13247)."""

from __future__ import annotations

from enum import Enum


class EnumCodingAgent(str, Enum):
    """The coding-agent CLI a ``ModelCodingAgentInvokeCommand`` invokes.

    The invoke effect dispatches on this field to Claude Code or Codex; both are
    selected per call and swappable by contract/overlay (no hardcoded default).
    """

    CLAUDE = "claude"
    CODEX = "codex"
