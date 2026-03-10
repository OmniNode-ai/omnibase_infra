# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Dependency injection registry for the setup Infisical effect node.

Ticket: OMN-3494
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_setup_infisical_effect.handlers.handler_infisical_full_setup import (
        HandlerInfisicalFullSetup,
    )

logger = logging.getLogger(__name__)

_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


class RegistryInfraSetupInfisicalEffect:
    """Registry for the Infisical setup effect node handlers."""

    HANDLER_KEY = "handler_infisical_full_setup"

    @staticmethod
    def _is_registered(handler_key: str) -> bool:
        return handler_key in _HANDLER_STORAGE

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register Infisical setup effect metadata with the container."""
        _PROTOCOL_METADATA[RegistryInfraSetupInfisicalEffect.HANDLER_KEY] = {
            "handler": "HandlerInfisicalFullSetup",
            "module": (
                "omnibase_infra.nodes.node_setup_infisical_effect"
                ".handlers.handler_infisical_full_setup"
            ),
            "description": "Handler for Infisical provision and seed operations",
            "capabilities": [
                "setup.infisical.provision",
                "setup.infisical.seed",
            ],
        }

    @staticmethod
    def register_handler(
        _container: ModelONEXContainer,
        handler: HandlerInfisicalFullSetup,
        handler_type: str = "full_setup",
    ) -> None:
        """Register a specific Infisical setup handler."""
        required_methods = ["initialize", "shutdown", "execute"]
        missing = [
            m for m in required_methods if not callable(getattr(handler, m, None))
        ]
        if missing:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="register_handler",
                target_name="infisical_setup_handler",
            )
            raise ProtocolConfigurationError(
                f"Handler missing required protocol methods: {missing}",
                context=context,
            )

        handler_key = f"{RegistryInfraSetupInfisicalEffect.HANDLER_KEY}.{handler_type}"

        if RegistryInfraSetupInfisicalEffect._is_registered(handler_key):
            logger.warning("Re-registering handler '%s'", handler_key)

        _HANDLER_STORAGE[handler_key] = handler

    @staticmethod
    def get_handler(
        _container: ModelONEXContainer,
        handler_type: str = "full_setup",
    ) -> object | None:
        """Retrieve a registered Infisical setup handler."""
        handler_key = f"{RegistryInfraSetupInfisicalEffect.HANDLER_KEY}.{handler_type}"
        return _HANDLER_STORAGE.get(handler_key)

    @staticmethod
    def clear() -> None:
        """Clear all registered handlers. Call in test teardown."""
        _HANDLER_STORAGE.clear()
        _PROTOCOL_METADATA.clear()


__all__: list[str] = ["RegistryInfraSetupInfisicalEffect"]
