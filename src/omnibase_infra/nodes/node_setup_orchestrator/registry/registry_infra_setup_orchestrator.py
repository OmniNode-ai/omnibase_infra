# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Dependency injection registry for the setup orchestrator node.

Ticket: OMN-3495
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_setup_orchestrator.handlers.handler_setup_orchestrator import (
        HandlerSetupOrchestrator,
    )

logger = logging.getLogger(__name__)

_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


class RegistryInfraSetupOrchestrator:
    """Registry for setup orchestrator node handlers."""

    HANDLER_KEY = "handler_setup_orchestrator"

    @staticmethod
    def _is_registered(handler_key: str) -> bool:
        return handler_key in _HANDLER_STORAGE

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register setup orchestrator metadata with the container."""
        _PROTOCOL_METADATA[RegistryInfraSetupOrchestrator.HANDLER_KEY] = {
            "handler": "HandlerSetupOrchestrator",
            "module": (
                "omnibase_infra.nodes.node_setup_orchestrator"
                ".handlers.handler_setup_orchestrator"
            ),
            "description": "Handler for setup orchestration workflow",
            "capabilities": [
                "setup.orchestrate",
                "setup.cloud.gate",
                "setup.preflight",
                "setup.provision",
                "setup.infisical",
                "setup.validate",
            ],
        }

    @staticmethod
    def register_handler(
        _container: ModelONEXContainer,
        handler: HandlerSetupOrchestrator,
        handler_type: str = "orchestrate",
    ) -> None:
        """Register a specific setup orchestrator handler."""
        required_methods = ["initialize", "shutdown", "execute", "handle"]
        missing = [
            m for m in required_methods if not callable(getattr(handler, m, None))
        ]
        if missing:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="register_handler",
                target_name="setup_orchestrator_handler",
            )
            raise ProtocolConfigurationError(
                f"Handler missing required protocol methods: {missing}",
                context=context,
            )

        handler_key = f"{RegistryInfraSetupOrchestrator.HANDLER_KEY}.{handler_type}"

        if RegistryInfraSetupOrchestrator._is_registered(handler_key):
            logger.warning("Re-registering handler '%s'", handler_key)

        _HANDLER_STORAGE[handler_key] = handler

    @staticmethod
    def get_handler(
        _container: ModelONEXContainer,
        handler_type: str = "orchestrate",
    ) -> object | None:
        """Retrieve a registered setup orchestrator handler."""
        handler_key = f"{RegistryInfraSetupOrchestrator.HANDLER_KEY}.{handler_type}"
        return _HANDLER_STORAGE.get(handler_key)

    @staticmethod
    def clear() -> None:
        """Clear all registered handlers. Call in test teardown."""
        _HANDLER_STORAGE.clear()
        _PROTOCOL_METADATA.clear()


__all__: list[str] = ["RegistryInfraSetupOrchestrator"]
