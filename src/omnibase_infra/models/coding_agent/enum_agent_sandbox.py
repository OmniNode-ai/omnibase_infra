# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Sandbox / write-mode for a coding-agent invocation (OMN-13247)."""

from __future__ import annotations

from enum import Enum


class EnumAgentSandbox(str, Enum):
    """The write posture OmniNode grants a coding-agent invocation.

    ``READ_ONLY`` agents must not edit the workspace; ``WORKSPACE_WRITE`` agents
    may. These are OmniNode-canonical semantics; the per-CLI permission-mode argv
    mapping (``codex -s ...`` / ``claude --permission-mode ...``) is deferred to
    post-Phase-0 once the exact per-version behavior is proven (see plan §5.7).
    """

    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
