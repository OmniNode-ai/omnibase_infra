# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle dispatch for the listener broker EFFECT transport (OMN-22379)."""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_runtime_listener_broker_request import (
    ModelRuntimeListenerBrokerRequest,
)
from omnibase_infra.models.model_runtime_listener_broker_response import (
    ModelRuntimeListenerBrokerResponse,
)
from omnibase_infra.models.model_runtime_listener_broker_server_io import (
    ModelRuntimeListenerBrokerServerIo,
)
from omnibase_infra.models.model_runtime_listener_registry import (
    ModelRuntimeListenerRegistry,
)


class HandlerRuntimeListenerBrokerServer:
    """Dispatch transport lifecycle to the broker node's injected I/O runtime."""

    def __init__(self, container: ModelRuntimeListenerRegistry) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelRuntimeListenerBrokerRequest
    ) -> ModelRuntimeListenerBrokerResponse:
        self.start()
        return ModelRuntimeListenerBrokerResponse(
            ok=True, op=request.op.value, pool=request.pool.value
        )

    def start(self) -> None:
        self._container.get_service(ModelRuntimeListenerBrokerServerIo).start()

    def stop(self) -> None:
        self._container.get_service(ModelRuntimeListenerBrokerServerIo).stop()


__all__: list[str] = ["HandlerRuntimeListenerBrokerServer"]
