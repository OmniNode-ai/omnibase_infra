# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""How the task-class authority resolved a prompt, as ``onex delegate`` reads it (OMN-19407)."""

from __future__ import annotations

from typing import Protocol

__all__ = ["ProtocolTaskTypeResolution"]


class ProtocolTaskTypeResolution(Protocol):
    """The authority's answer: the class, how it was decided, and why."""

    @property
    def task_type(self) -> str: ...

    @property
    def resolution(self) -> str: ...

    @property
    def reason(self) -> str: ...
