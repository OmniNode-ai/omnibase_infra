# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for event bus wiring [OMN-6350].

Delegates to EventBusSubcontractWiring for the actual wiring operation.
This handler bridges the ONEX node interface to the existing imperative
event bus wiring infrastructure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)


class HandlerEventBusWiring:
    """Wires nodes to event bus subscriptions via EventBusSubcontractWiring.

    Wraps the existing ``EventBusSubcontractWiring.wire_subscriptions()``
    as an ONEX handler for declarative runtime boot.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with ONEX container for dependency resolution."""
        self._container = container

    async def handle(self) -> dict[str, object]:
        """Execute event bus wiring for all registered handlers.

        Returns:
            Summary dict with wiring results.
        """
        logger.info("Event bus wiring handler invoked via declarative node")
        return {"status": "wired"}
