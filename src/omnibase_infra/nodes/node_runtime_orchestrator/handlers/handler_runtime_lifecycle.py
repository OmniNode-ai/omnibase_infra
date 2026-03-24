# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for runtime boot sequence orchestration [OMN-6351].

Coordinates the 4-step sequential boot:
    1. contract_loader -- scan filesystem for contracts
    2. contract_registry -- project contracts to registry
    3. node_graph -- instantiate node graph
    4. event_bus_wiring -- wire nodes to Kafka

Step callables are injected through the constructor for testability.
Fail-fast: if any step raises, subsequent steps are not executed.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

BootStep = Callable[[], Awaitable[None]]

_STEP_NAMES: tuple[str, ...] = (
    "contract_loader",
    "contract_registry",
    "node_graph",
    "event_bus_wiring",
)


class HandlerRuntimeLifecycle:
    """Orchestrates the 4-step runtime boot sequence.

    Steps are injected as callables for testability.
    Sequential execution with fail-fast semantics.
    """

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        *,
        steps: tuple[BootStep, ...] | None = None,
    ) -> None:
        """Initialize with either container or explicit step callables.

        Args:
            container: ONEX container (stored for future resolution).
            steps: Optional explicit step callables in boot order.
                If provided, used directly. If None, steps must be
                provided via execute_startup(steps=...).
        """
        self._container = container
        self._steps = steps

    async def execute_startup(
        self,
        steps: tuple[BootStep, ...] | None = None,
    ) -> None:
        """Execute the boot sequence in order.

        Args:
            steps: Optional override for step callables.

        Raises:
            Any exception from a step -- fail-fast, no subsequent steps run.
        """
        active_steps = steps or self._steps
        if not active_steps:
            msg = "No boot steps provided to HandlerRuntimeLifecycle"
            raise ValueError(msg)

        for step_name, step_callable in zip(_STEP_NAMES, active_steps, strict=False):
            logger.info("Runtime boot step: %s", step_name)
            await step_callable()
            logger.info("Runtime boot step complete: %s", step_name)

        logger.info(
            "Runtime boot sequence complete (all %d steps)",
            len(active_steps),
        )
