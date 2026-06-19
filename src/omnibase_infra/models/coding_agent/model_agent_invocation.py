# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolved coding-agent CLI invocation."""

from __future__ import annotations


class ModelAgentInvocation:
    """The fully-resolved argv + optional stdin for one coding-agent invocation."""

    __slots__ = ("argv", "stdin")

    def __init__(self, *, argv: list[str], stdin: str | None) -> None:
        self.argv = argv
        self.stdin = stdin


__all__ = ["ModelAgentInvocation"]
