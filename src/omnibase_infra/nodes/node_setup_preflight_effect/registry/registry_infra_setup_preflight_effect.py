# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Dependency injection registry for the setup preflight effect node.

Ticket: OMN-3492
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_setup_preflight_effect.handlers.handler_preflight_check import (
        HandlerPreflightCheck,
    )

logger = logging.getLogger(__name__)

_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


class RegistryInfraSetupPreflightEffect:
    """Registry for setup preflight effect node handlers."""

    HANDLER_KEY = "handler_preflight_check"

    @staticmethod
    def _is_registered(handler_key: str) -> bool:
        return handler_key in _HANDLER_STORAGE

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register preflight effect metadata with the container."""
        _PROTOCOL_METADATA[RegistryInfraSetupPreflightEffect.HANDLER_KEY] = {
            "handler": "HandlerPreflightCheck",
            "module": "omnibase_infra.nodes.node_setup_preflight_effect.handlers.handler_preflight_check",
            "description": "Handler for preflight check operations",
            "capabilities": [
                "preflight.check",
            ],
        }

    @staticmethod
    def register_handler(
        _container: ModelONEXContainer,
        handler: HandlerPreflightCheck,
        handler_type: str = "check",
    ) -> None:
        """Register a specific preflight handler."""
        required_methods = ["initialize", "shutdown", "execute"]
        missing = [
            m for m in required_methods if not callable(getattr(handler, m, None))
        ]
        if missing:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="register_handler",
                target_name="preflight_handler",
            )
            raise ProtocolConfigurationError(
                f"Handler missing required protocol methods: {missing}",
                context=context,
            )

        handler_key = f"{RegistryInfraSetupPreflightEffect.HANDLER_KEY}.{handler_type}"

        if RegistryInfraSetupPreflightEffect._is_registered(handler_key):
            logger.warning("Re-registering handler '%s'", handler_key)

        _HANDLER_STORAGE[handler_key] = handler

    @staticmethod
    def get_handler(
        _container: ModelONEXContainer,
        handler_type: str = "check",
    ) -> object | None:
        """Retrieve a registered preflight handler."""
        handler_key = f"{RegistryInfraSetupPreflightEffect.HANDLER_KEY}.{handler_type}"
        return _HANDLER_STORAGE.get(handler_key)

    @staticmethod
    def clear() -> None:
        """Clear all registered handlers. Call in test teardown."""
        _HANDLER_STORAGE.clear()
        _PROTOCOL_METADATA.clear()


__all__: list[str] = ["RegistryInfraSetupPreflightEffect"]
