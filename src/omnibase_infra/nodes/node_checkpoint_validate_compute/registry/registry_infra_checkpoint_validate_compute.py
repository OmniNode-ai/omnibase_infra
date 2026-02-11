# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Dependency injection registry for the checkpoint validate compute node.

Ticket: OMN-2143
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_checkpoint_validate_compute.handlers.handler_checkpoint_validate import (
        HandlerCheckpointValidate,
    )

logger = logging.getLogger(__name__)

_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


class RegistryInfraCheckpointValidateCompute:
    """Registry for checkpoint validate compute node handler."""

    HANDLER_KEY = "handler_checkpoint_validate"

    @staticmethod
    def _is_registered(handler_key: str) -> bool:
        return handler_key in _HANDLER_STORAGE

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register checkpoint validate metadata."""
        _PROTOCOL_METADATA[RegistryInfraCheckpointValidateCompute.HANDLER_KEY] = {
            "handler": "HandlerCheckpointValidate",
            "module": "omnibase_infra.nodes.node_checkpoint_validate_compute.handlers.handler_checkpoint_validate",
            "description": "Handler for checkpoint structural validation",
            "capabilities": ["checkpoint.validate"],
        }

    @staticmethod
    def register_handler(
        _container: ModelONEXContainer,
        handler: HandlerCheckpointValidate,
    ) -> None:
        """Register a checkpoint validate handler."""
        required_methods = ["initialize", "shutdown", "execute"]
        missing = [
            m for m in required_methods if not callable(getattr(handler, m, None))
        ]
        if missing:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="register_handler",
                target_name="checkpoint_validate_handler",
            )
            raise ProtocolConfigurationError(
                f"Handler missing required protocol methods: {missing}",
                context=context,
            )

        handler_key = f"{RegistryInfraCheckpointValidateCompute.HANDLER_KEY}.default"

        if RegistryInfraCheckpointValidateCompute._is_registered(handler_key):
            logger.warning("Re-registering handler '%s'", handler_key)

        _HANDLER_STORAGE[handler_key] = handler

    @staticmethod
    def get_handler(
        _container: ModelONEXContainer,
    ) -> object | None:
        """Retrieve a registered checkpoint validate handler."""
        handler_key = f"{RegistryInfraCheckpointValidateCompute.HANDLER_KEY}.default"
        return _HANDLER_STORAGE.get(handler_key)

    @staticmethod
    def clear() -> None:
        """Clear all registered handlers. Call in test teardown."""
        _HANDLER_STORAGE.clear()
        _PROTOCOL_METADATA.clear()


__all__: list[str] = ["RegistryInfraCheckpointValidateCompute"]
