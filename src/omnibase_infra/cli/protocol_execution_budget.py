# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One task class's declared budget, as ``onex delegate`` reads it (OMN-19407)."""

from __future__ import annotations

from typing import Protocol

__all__ = ["ProtocolExecutionBudget"]


class ProtocolExecutionBudget(Protocol):
    """One class's declared handler ceiling and terminal delivery margin."""

    @property
    def task_class_timeout_ceiling_seconds(self) -> int: ...

    @property
    def terminal_delivery_margin_seconds(self) -> int: ...
