# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Dependency injection registry for the checkpoint effect node.

Ticket: OMN-2143
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_checkpoint_effect.handlers.handler_checkpoint_list import (
        HandlerCheckpointList,
    )
    from omnibase_infra.nodes.node_checkpoint_effect.handlers.handler_checkpoint_read import (
        HandlerCheckpointRead,
    )
    from omnibase_infra.nodes.node_checkpoint_effect.handlers.handler_checkpoint_write import (
        HandlerCheckpointWrite,
    )

logger = logging.getLogger(__name__)

_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


class RegistryInfraCheckpointEffect:
    """Registry for checkpoint effect node handlers."""

    HANDLER_KEY = "handler_checkpoint"

    @staticmethod
    def _is_registered(handler_key: str) -> bool:
        return handler_key in _HANDLER_STORAGE

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register checkpoint effect metadata with the container."""
        _PROTOCOL_METADATA[RegistryInfraCheckpointEffect.HANDLER_KEY] = {
            "handler": "HandlerCheckpointWrite",
            "module": "omnibase_infra.nodes.node_checkpoint_effect.handlers.handler_checkpoint_write",
            "description": "Handler for checkpoint write operations",
            "capabilities": [
                "checkpoint.write",
                "checkpoint.read",
                "checkpoint.list",
            ],
        }

    @staticmethod
    def register_handler(
        _container: ModelONEXContainer,
        handler: HandlerCheckpointWrite | HandlerCheckpointRead | HandlerCheckpointList,
        handler_type: str = "write",
    ) -> None:
        """Register a specific checkpoint handler."""
        required_methods = ["initialize", "shutdown", "execute"]
        missing = [
            m for m in required_methods if not callable(getattr(handler, m, None))
        ]
        if missing:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="register_handler",
                target_name="checkpoint_handler",
            )
            raise ProtocolConfigurationError(
                f"Handler missing required protocol methods: {missing}",
                context=context,
            )

        handler_key = f"{RegistryInfraCheckpointEffect.HANDLER_KEY}.{handler_type}"

        if RegistryInfraCheckpointEffect._is_registered(handler_key):
            logger.warning("Re-registering handler '%s'", handler_key)

        _HANDLER_STORAGE[handler_key] = handler

    @staticmethod
    def get_handler(
        _container: ModelONEXContainer,
        handler_type: str = "write",
    ) -> object | None:
        """Retrieve a registered checkpoint handler."""
        handler_key = f"{RegistryInfraCheckpointEffect.HANDLER_KEY}.{handler_type}"
        return _HANDLER_STORAGE.get(handler_key)

    @staticmethod
    def clear() -> None:
        """Clear all registered handlers. Call in test teardown."""
        _HANDLER_STORAGE.clear()
        _PROTOCOL_METADATA.clear()


__all__: list[str] = ["RegistryInfraCheckpointEffect"]
