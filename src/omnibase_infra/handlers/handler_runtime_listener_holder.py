# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request dispatch for the runtime listener holder EFFECT node (OMN-22379).

The flat handler owns operation routing. Socket, descriptor, environment and
filesystem effects are supplied by the node's I/O runtime through the container.
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_runtime_listener_fd_store import (
    ModelRuntimeListenerFdStore,
)
from omnibase_infra.models.model_runtime_listener_holder_io import (
    ModelRuntimeListenerHolderIo,
)
from omnibase_infra.models.model_runtime_listener_holder_request import (
    ModelRuntimeListenerHolderRequest,
)
from omnibase_infra.models.model_runtime_listener_holder_result import (
    ModelRuntimeListenerHolderResult,
)
from omnibase_infra.models.model_runtime_listener_registry import (
    ModelRuntimeListenerRegistry,
)


class HandlerRuntimeListenerHolder:
    """Dispatch holder operations to the injected EFFECT runtime."""

    def __init__(self, container: ModelRuntimeListenerRegistry) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    @property
    def fd_store(self) -> ModelRuntimeListenerFdStore:
        return self._container.get_service(ModelRuntimeListenerFdStore)

    def handle(
        self, request: ModelRuntimeListenerHolderRequest
    ) -> ModelRuntimeListenerHolderResult:
        io = self._container.get_service(ModelRuntimeListenerHolderIo)
        match request.operation:
            case "install_bind":
                return io.install_bind(request)
            case "recover":
                return io.recover()
            case "reboot_rebind":
                return io.reboot_rebind(request)
            case "prepare_reexec":
                return io.prepare_reexec()
            case "reexec":
                return io.reexec(request)
            case "drop":
                return io.drop()
        raise ValueError(f"unknown holder operation {request.operation!r}")

    def prepare_for_reexec(self) -> ModelRuntimeListenerHolderResult:
        """Prepare-only remap; the live agent uses the transactional reexec operation."""
        return self.handle(
            ModelRuntimeListenerHolderRequest(operation="prepare_reexec")
        )


__all__: list[str] = ["HandlerRuntimeListenerHolder"]
