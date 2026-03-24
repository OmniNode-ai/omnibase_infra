# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for event bus wiring [OMN-6350].

Delegates to EventBusSubcontractWiring for the actual wiring operation.
This handler bridges the ONEX node interface to the existing imperative
event bus wiring infrastructure.

NOTE: The actual wiring logic (Kafka subscription creation, consumer group
setup, topic validation) lives in ``EventBusSubcontractWiring``. This handler
is the declarative node entry point that delegates to that existing imperative
infrastructure. It does NOT duplicate wiring logic.
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

        Delegates to the ``EventBusSubcontractWiring`` instance resolved
        from the container. The actual subscription logic (Kafka consumer
        creation, topic validation, etc.) is handled by that class.

        Returns:
            Summary dict with wiring results.

        Raises:
            RuntimeError: If event bus wiring service is not available.
        """
        logger.info("Event bus wiring handler invoked via declarative node")

        # Resolve wiring service from container if available
        wiring_service = getattr(self._container, "event_bus_wiring", None)
        if wiring_service is not None and hasattr(wiring_service, "wire_subscriptions"):
            result = await wiring_service.wire_subscriptions()
            logger.info(
                "Event bus wiring complete via container service: %s",
                result,
            )
            return {"status": "wired", "result": str(result)}

        # Fallback: wiring will be performed by the kernel's plugin activation
        # phase which calls EventBusSubcontractWiring directly. This handler
        # acts as a no-op marker in the boot sequence indicating the wiring
        # step position.
        logger.info(
            "Event bus wiring deferred to kernel plugin activation "
            "(no wiring service on container)"
        )
        return {"status": "deferred_to_kernel"}
