# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for Registration Storage Node Dependencies.

This module provides RegistryInfraRegistrationStorage, which registers
dependencies for the NodeRegistrationStorageEffect node.

Architecture:
    The registry follows ONEX container-based dependency injection:
    - Registers protocol implementations with ModelONEXContainer
    - Supports pluggable handler backends (PostgreSQL, mock for testing)
    - Enables runtime handler selection based on configuration

    Registration is typically called during application bootstrap.

Related:
    - NodeRegistrationStorageEffect: Effect node that uses these dependencies
    - ProtocolRegistrationStorageHandler: Protocol for storage backends
    - ModelONEXContainer: ONEX dependency injection container
"""

from __future__ import annotations

__all__ = ["RegistryInfraRegistrationStorage"]

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

    from omnibase_infra.nodes.node_registration_storage_effect.protocols import (
        ProtocolRegistrationStorageHandler,
    )


class RegistryInfraRegistrationStorage:
    """Registry for registration storage node dependencies.

    Registers handler protocols and implementations with the ONEX container.
    Supports pluggable backends through handler registration.

    Usage:
        .. code-block:: python

            from omnibase_core.models.container import ModelONEXContainer
            from omnibase_infra.nodes.node_registration_storage_effect.registry import (
                RegistryInfraRegistrationStorage,
            )

            # Create container
            container = ModelONEXContainer()

            # Register dependencies
            RegistryInfraRegistrationStorage.register(container)

            # Optionally register a specific handler
            RegistryInfraRegistrationStorage.register_handler(
                container,
                handler=postgres_handler,
            )

    Note:
        This registry does NOT instantiate handlers. Handlers must be
        created externally with their specific dependencies (connection
        pools, configs) and then registered via register_handler().
    """

    # Protocol key for container registration
    PROTOCOL_KEY = "protocol_registration_storage_handler"

    # Default handler type when multiple are registered
    DEFAULT_HANDLER_TYPE = "postgresql"

    @staticmethod
    def register(container: ModelONEXContainer) -> None:
        """Register registration storage dependencies with the container.

        Registers the protocol key for later handler binding. This method
        sets up the infrastructure but does not bind a specific handler.

        Args:
            container: ONEX dependency injection container.

        Example:
            >>> from omnibase_core.models.container import ModelONEXContainer
            >>> container = ModelONEXContainer()
            >>> RegistryInfraRegistrationStorage.register(container)
        """
        # Register protocol metadata for discovery
        # Actual handler binding happens via register_handler()
        if container.service_registry is not None:
            container.service_registry[
                RegistryInfraRegistrationStorage.PROTOCOL_KEY
            ] = {
                "protocol": "ProtocolRegistrationStorageHandler",
                "module": "omnibase_infra.nodes.node_registration_storage_effect.protocols",
                "description": "Protocol for registration storage backends",
                "pluggable": True,
                "implementations": ["postgresql", "mock"],
            }

    @staticmethod
    def register_handler(
        container: ModelONEXContainer,
        handler: ProtocolRegistrationStorageHandler,
    ) -> None:
        """Register a specific storage handler with the container.

        Binds a concrete handler implementation to the protocol key.
        The handler must implement ProtocolRegistrationStorageHandler.

        Args:
            container: ONEX dependency injection container.
            handler: Handler implementation to register.

        Raises:
            TypeError: If handler does not implement ProtocolRegistrationStorageHandler.

        Example:
            >>> from omnibase_infra.handlers.registration_storage import (
            ...     HandlerPostgresRegistrationStorage,
            ... )
            >>> handler = HandlerPostgresRegistrationStorage(pool, config)
            >>> RegistryInfraRegistrationStorage.register_handler(container, handler)
        """
        # Import at runtime for isinstance check (protocol is @runtime_checkable)
        from omnibase_infra.nodes.node_registration_storage_effect.protocols import (
            ProtocolRegistrationStorageHandler,
        )

        if not isinstance(handler, ProtocolRegistrationStorageHandler):
            raise TypeError(
                f"Handler must implement ProtocolRegistrationStorageHandler, "
                f"got {type(handler).__name__}"
            )

        if container.service_registry is None:
            return

        handler_key = (
            f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.{handler.handler_type}"
        )
        container.service_registry[handler_key] = handler

        # Also register as default if it matches the default type
        if (
            handler.handler_type
            == RegistryInfraRegistrationStorage.DEFAULT_HANDLER_TYPE
        ):
            container.service_registry[
                RegistryInfraRegistrationStorage.PROTOCOL_KEY + ".default"
            ] = handler

    @staticmethod
    def get_handler(
        container: ModelONEXContainer,
        handler_type: str | None = None,
    ) -> ProtocolRegistrationStorageHandler | None:
        """Retrieve a registered storage handler from the container.

        Args:
            container: ONEX dependency injection container.
            handler_type: Specific handler type to retrieve. If None, returns default.

        Returns:
            The registered handler, or None if not found.

        Example:
            >>> handler = RegistryInfraRegistrationStorage.get_handler(
            ...     container,
            ...     handler_type="postgresql",
            ... )
        """
        if container.service_registry is None:
            return None

        if handler_type is not None:
            handler_key = (
                f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.{handler_type}"
            )
        else:
            handler_key = RegistryInfraRegistrationStorage.PROTOCOL_KEY + ".default"

        result = container.service_registry.get(handler_key)
        return cast("ProtocolRegistrationStorageHandler | None", result)
