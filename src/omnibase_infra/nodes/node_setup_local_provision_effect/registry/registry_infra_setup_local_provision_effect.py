# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Dependency injection registry for the setup local provision effect node.

Ticket: OMN-3493
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


class RegistryInfraSetupLocalProvisionEffect:
    """Registry for setup local provision effect node handlers."""

    HANDLER_KEY = "handler_local_provision"

    @staticmethod
    def _is_registered(handler_key: str) -> bool:
        """Return True if a handler with the given key has been registered."""
        return handler_key in _HANDLER_STORAGE

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register local provision effect metadata with the container."""
        _PROTOCOL_METADATA[RegistryInfraSetupLocalProvisionEffect.HANDLER_KEY] = {
            "handler": "HandlerLocalProvision",
            "module": "omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_provision",
            "description": "Handler for local Docker Compose provisioning operations",
            "capabilities": [
                "local_provision.start",
                "local_provision.stop",
                "local_provision.status",
            ],
        }

    @staticmethod
    def register_handler(
        _container: ModelONEXContainer,
        handler: object,
        handler_type: str = "provision",
    ) -> None:
        """Register a specific local provision handler.

        Args:
            _container: DI container (unused, reserved for future use).
            handler: Handler instance to register (HandlerLocalProvision,
                HandlerLocalTeardown, or HandlerLocalStatus).
            handler_type: One of "provision", "teardown", or "status".
        """
        required_methods = ["initialize", "shutdown", "execute"]
        missing = [
            m for m in required_methods if not callable(getattr(handler, m, None))
        ]
        if missing:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="register_handler",
                target_name="local_provision_handler",
            )
            raise ProtocolConfigurationError(
                f"Handler missing required protocol methods: {missing}",
                context=context,
            )

        handler_key = (
            f"{RegistryInfraSetupLocalProvisionEffect.HANDLER_KEY}.{handler_type}"
        )

        if RegistryInfraSetupLocalProvisionEffect._is_registered(handler_key):
            logger.warning("Re-registering handler '%s'", handler_key)

        _HANDLER_STORAGE[handler_key] = handler

    @staticmethod
    def get_handler(
        _container: ModelONEXContainer,
        handler_type: str = "provision",
    ) -> object | None:
        """Retrieve a registered local provision handler.

        Args:
            _container: DI container (unused, reserved for future use).
            handler_type: One of "provision", "teardown", or "status".

        Returns:
            The registered handler or None if not found.
        """
        handler_key = (
            f"{RegistryInfraSetupLocalProvisionEffect.HANDLER_KEY}.{handler_type}"
        )
        return _HANDLER_STORAGE.get(handler_key)

    @staticmethod
    def clear() -> None:
        """Clear all registered handlers. Call in test teardown."""
        _HANDLER_STORAGE.clear()
        _PROTOCOL_METADATA.clear()


__all__: list[str] = ["RegistryInfraSetupLocalProvisionEffect"]
