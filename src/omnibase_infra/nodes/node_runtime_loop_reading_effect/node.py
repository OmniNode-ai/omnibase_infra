# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative effect shell for the runtime event-loop reading (OMN-26249)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeRuntimeLoopReadingEffect(NodeEffect):
    """Declarative effect node for a runtime member's per-window loop reading.

    Handlers (``omnibase_infra.handlers``):
        - ``HandlerRuntimeLoopLagSampler``: samples the event loop's lateness
          and closes a reading window into a ``ModelRuntimeLoopReading``.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the runtime loop reading effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeRuntimeLoopReadingEffect"]
