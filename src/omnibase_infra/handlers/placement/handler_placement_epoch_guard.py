# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Write-target epoch guard (OMN-21421).

The model's ``Deliver`` with ``WriteGuard``: a target accepts a command only
when its ``placement_epoch`` is at least the highest epoch it has accepted for
that node, so a write a fenced primary sent before its deadline cannot land
after a newer primary's.
"""

from __future__ import annotations

from omnibase_infra.handlers.placement.model_placement_command import (
    ModelPlacementCommand,
)


class HandlerPlacementEpochGuard:
    """Per-node highest accepted epoch; refuses stale commands."""

    def __init__(self) -> None:
        self._highest: dict[str, int] = {}

    def highest(self, node_name: str) -> int:
        return self._highest.get(node_name, 0)

    def accept(self, command: ModelPlacementCommand) -> bool:
        if command.placement_epoch < self.highest(command.node_key):
            return False
        self._highest[command.node_key] = command.placement_epoch
        return True


__all__ = ["HandlerPlacementEpochGuard"]
