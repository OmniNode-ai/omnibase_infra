# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's service container for the runtime listener handlers (OMN-22379).

The agent has no ``ModelONEXContainer``; this resolves the same way
(``get_service(type)``) from the instances it registers. ``start`` is the one
entry the agent calls: it returns ``None`` until a unit hands over descriptors
(``LISTEN_FDS``) or declares a handover directory (``DEV_RUNTIME_HANDOVER_DIR``),
so nothing is imported, bound or created before an install.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, PrivateAttr

from omnibase_infra.models.model_runtime_listener_broker_port import (
    ModelRuntimeListenerBrokerPort,
)
from omnibase_infra.models.model_runtime_listener_endpoint import (
    ModelRuntimeListenerEndpoint,
)
from omnibase_infra.models.model_runtime_listener_environment import (
    ModelRuntimeListenerEnvironment,
)
from omnibase_infra.models.model_runtime_listener_fd_store import (
    ModelRuntimeListenerFdStore,
)
from omnibase_infra.models.model_runtime_listener_holder_request import (
    ModelRuntimeListenerHolderRequest,
)
from omnibase_infra.models.model_runtime_listener_holder_result import (
    ModelRuntimeListenerHolderResult,
)
from omnibase_infra.models.model_runtime_listener_session_table import (
    ModelRuntimeListenerSessionTable,
)

logger = logging.getLogger(__name__)


class ModelRuntimeListenerRegistry(BaseModel):
    """Registered collaborators of the listener holder, broker and socket server."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    _services: dict[type, object] = PrivateAttr(default_factory=dict)

    def register[T](self, protocol_type: type[T], instance: T) -> T:
        """Register ``instance`` under ``protocol_type`` and return it."""
        self._services[protocol_type] = instance
        return instance

    def get_service[T](self, protocol_type: type[T]) -> T:
        """Return the instance registered for ``protocol_type``."""
        instance = self._services.get(protocol_type)
        if not isinstance(instance, protocol_type):
            raise LookupError(f"no service registered for {protocol_type.__name__}")
        return instance

    @classmethod
    def assemble(
        cls,
        environ: MutableMapping[str, str],
        socket_path: Path | None = None,
    ) -> Self:
        """Register the collaborators and the broker; ``socket_path`` adds the endpoint."""
        from omnibase_infra.handlers.handler_runtime_listener_broker import (
            HandlerRuntimeListenerBroker,
        )

        services = cls()
        services.register(ModelRuntimeListenerFdStore, ModelRuntimeListenerFdStore())
        services.register(
            ModelRuntimeListenerSessionTable, ModelRuntimeListenerSessionTable()
        )
        services.register(
            ModelRuntimeListenerEnvironment,
            ModelRuntimeListenerEnvironment(values=environ),
        )
        if socket_path is not None:
            services.register(
                ModelRuntimeListenerEndpoint,
                ModelRuntimeListenerEndpoint(socket_path=socket_path),
            )
        services.register(
            ModelRuntimeListenerBrokerPort,
            ModelRuntimeListenerBrokerPort(
                serve=HandlerRuntimeListenerBroker(services).handle
            ),
        )
        return services

    @classmethod
    def start(cls, environ: MutableMapping[str, str] | None = None) -> Self | None:
        """Recover or rebind the listeners and serve the broker; ``None`` while inert."""
        env = os.environ if environ is None else environ
        handover_dir = env.get("DEV_RUNTIME_HANDOVER_DIR", "")
        if not env.get("LISTEN_FDS") and not handover_dir:
            return None
        from omnibase_infra.handlers.handler_runtime_listener_broker_server import (
            HandlerRuntimeListenerBrokerServer,
        )
        from omnibase_infra.handlers.handler_runtime_listener_holder import (
            HandlerRuntimeListenerHolder,
        )

        services = cls.assemble(
            env, Path(handover_dir) / "broker.sock" if handover_dir else None
        )
        holder = HandlerRuntimeListenerHolder(services)
        result = holder.handle(ModelRuntimeListenerHolderRequest(operation="recover"))
        if not holder.fd_store.pools() and handover_dir:
            result = holder.handle(
                ModelRuntimeListenerHolderRequest(
                    operation="reboot_rebind",
                    install_record_path=str(Path(handover_dir) / "install.json"),
                )
            )
        logger.info(
            "runtime listener holder: %s pools=%s", result.outcome, result.pools
        )
        if holder.fd_store.pools() and handover_dir:
            HandlerRuntimeListenerBrokerServer(services).start()
        return services

    def prepare_for_reexec(self) -> ModelRuntimeListenerHolderResult:
        """Remap the listeners; the agent calls this immediately before ``os.execv``."""
        from omnibase_infra.handlers.handler_runtime_listener_holder import (
            HandlerRuntimeListenerHolder,
        )

        return HandlerRuntimeListenerHolder(self).prepare_for_reexec()


__all__: list[str] = ["ModelRuntimeListenerRegistry"]
