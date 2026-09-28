# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Thin contract handler for a signed execution graph read command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.execution_graph_replay.model_execution_graph_request import (
    ModelExecutionGraphRequest,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.runtime.dispatch_envelope_context import (
    current_dispatch_envelope,
)
from omnibase_infra.runtime.execution_graph_read_command_handler import (
    ExecutionGraphReadCommandExecutor,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

HANDLER_ID_EXECUTION_GRAPH_READ = "execution-graph-read-handler"


class HandlerExecutionGraphRead:
    """Delegate one routed request to an explicitly composed read executor."""

    def __init__(
        self,
        container: ModelONEXContainer,
        executor: ExecutionGraphReadCommandExecutor,
    ) -> None:
        if not isinstance(executor, ExecutionGraphReadCommandExecutor):
            raise TypeError("execution graph read requires a composed executor")
        self._container = container
        self._executor = executor

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelExecutionGraphRequest
    ) -> ModelHandlerOutput[None]:
        envelope = current_dispatch_envelope()
        if envelope is None:
            raise RuntimeError(
                "execution graph read requires a typed dispatch envelope"
            )
        await self._executor.handle(request)
        return ModelHandlerOutput.for_effect(
            input_envelope_id=envelope.envelope_id,
            correlation_id=request.correlation_id,
            handler_id=HANDLER_ID_EXECUTION_GRAPH_READ,
        )


__all__ = ["HandlerExecutionGraphRead", "HANDLER_ID_EXECUTION_GRAPH_READ"]
