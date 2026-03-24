# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for runtime boot sequence orchestration [OMN-6351].

Coordinates the 4-step sequential boot:
    1. ProtocolContractLoader -- scan filesystem for contracts
    2. ProtocolContractRegistry -- project contracts to registry
    3. ProtocolNodeGraph -- instantiate node graph
    4. ProtocolEventBusWiring -- wire nodes to Kafka

Step dependencies are resolved via container.get_service() at execution
time, not stored as private attributes. Fail-fast: if any step raises,
subsequent steps are not executed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

_BOOT_STEPS: tuple[str, ...] = (
    "ProtocolContractLoader",
    "ProtocolContractRegistry",
    "ProtocolNodeGraph",
    "ProtocolEventBusWiring",
)


class HandlerRuntimeLifecycle:
    """Orchestrates the 4-step runtime boot sequence.

    Each step is resolved from the container at execution time.
    Sequential execution with fail-fast semantics.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with ONEX container for step resolution."""
        self._container = container

    async def execute_startup(self) -> None:
        """Execute the 4-step boot sequence in order.

        Raises:
            Any exception from a step -- fail-fast, no subsequent steps run.
        """
        for step_name in _BOOT_STEPS:
            logger.info("Runtime boot step: %s", step_name)
            step_callable = self._container.get_service(step_name)
            await step_callable()
            logger.info("Runtime boot step complete: %s", step_name)

        logger.info("Runtime boot sequence complete (all %d steps)", len(_BOOT_STEPS))
