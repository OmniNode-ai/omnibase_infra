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
    - ProtocolRegistrationPersistence: Protocol for storage backends
    - ModelONEXContainer: ONEX dependency injection container

Note:
    This registry uses a module-level dict for handler storage because the
    ServiceRegistry in omnibase_core v1.0 doesn't support dict-style access
    or string-keyed multi-handler routing. The handlers are still validated
    against the protocol but stored separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_registration_storage_effect.protocols import (
        ProtocolRegistrationPersistence,
    )

__all__ = ["RegistryInfraRegistrationStorage"]

# Module-level storage for handlers and metadata
# ServiceRegistry in v1.0 doesn't support dict-style access needed for
# multi-handler routing (e.g., "postgresql", "mock" handler types)
_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


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
    # Aligned with protocol name: ProtocolRegistrationPersistence
    PROTOCOL_KEY = "protocol_registration_persistence"

    # Default handler type when multiple are registered
    DEFAULT_HANDLER_TYPE = "postgresql"

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register registration storage dependencies with the container.

        Registers the protocol key for later handler binding. This method
        sets up the infrastructure but does not bind a specific handler.

        Args:
            _container: ONEX dependency injection container. Currently unused
                because ServiceRegistry v1.0 doesn't support dict-style access
                for multi-handler routing. The parameter is retained for API
                consistency with other registry methods and future migration
                when ServiceRegistry supports the required access patterns.

        Example:
            >>> from omnibase_core.models.container import ModelONEXContainer
            >>> container = ModelONEXContainer()
            >>> RegistryInfraRegistrationStorage.register(container)
        """
        # Register protocol metadata for discovery
        # Actual handler binding happens via register_handler()
        # Note: Uses module-level storage since ServiceRegistry v1.0 doesn't
        # support dict-style access for multi-handler routing
        _PROTOCOL_METADATA[RegistryInfraRegistrationStorage.PROTOCOL_KEY] = {
            "protocol": "ProtocolRegistrationPersistence",
            "module": "omnibase_infra.nodes.node_registration_storage_effect.protocols",
            "description": "Protocol for registration storage backends",
            "pluggable": True,
            "implementations": ["postgresql", "mock"],
        }

    @staticmethod
    def register_handler(
        _container: ModelONEXContainer,
        handler: ProtocolRegistrationPersistence,
    ) -> None:
        """Register a specific storage handler with the container.

        Binds a concrete handler implementation to the protocol key.
        The handler must implement ProtocolRegistrationPersistence.

        Args:
            _container: ONEX dependency injection container. Currently unused
                because ServiceRegistry v1.0 doesn't support dict-style access
                for multi-handler routing. The parameter is retained for API
                consistency and future migration when ServiceRegistry supports
                the required access patterns.
            handler: Handler implementation to register.

        Raises:
            TypeError: If handler does not implement ProtocolRegistrationPersistence.

        Example:
            >>> from omnibase_infra.handlers.registration_storage import (
            ...     HandlerRegistrationStoragePostgres,
            ... )
            >>> handler = HandlerRegistrationStoragePostgres(container, dsn="postgresql://...")
            >>> RegistryInfraRegistrationStorage.register_handler(container, handler)
        """
        # Import at runtime for isinstance check (protocol is @runtime_checkable)
        from omnibase_infra.nodes.node_registration_storage_effect.protocols import (
            ProtocolRegistrationPersistence,
        )

        # NOTE: isinstance() is intentionally used here instead of duck typing for:
        # 1. Fail-fast validation: Immediately reject invalid handlers at registration
        #    time rather than discovering missing methods at runtime during operations
        # 2. Type safety: The @runtime_checkable decorator enables structural subtyping
        #    checks that verify all required Protocol methods exist
        # 3. Clear error messages: TypeError with specific protocol name aids debugging
        # Duck typing (hasattr checks) would defer validation to method call time,
        # making it harder to diagnose misconfigured handlers.
        # See: conftest.py "Protocol Compliance Strategy" for when to use each approach.
        if not isinstance(handler, ProtocolRegistrationPersistence):
            raise TypeError(
                f"Handler must implement ProtocolRegistrationPersistence, "
                f"got {type(handler).__name__}"
            )

        # Note: Uses module-level storage since ServiceRegistry v1.0 doesn't
        # support dict-style access for multi-handler routing
        handler_key = (
            f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.{handler.handler_type}"
        )
        _HANDLER_STORAGE[handler_key] = handler

        # Also register as default if it matches the default type
        if (
            handler.handler_type
            == RegistryInfraRegistrationStorage.DEFAULT_HANDLER_TYPE
        ):
            _HANDLER_STORAGE[
                RegistryInfraRegistrationStorage.PROTOCOL_KEY + ".default"
            ] = handler

    @staticmethod
    def get_handler(
        _container: ModelONEXContainer,
        handler_type: str | None = None,
    ) -> ProtocolRegistrationPersistence | None:
        """Retrieve a registered storage handler from the container.

        Args:
            _container: ONEX dependency injection container. Currently unused
                because ServiceRegistry v1.0 doesn't support dict-style access
                for multi-handler routing. The parameter is retained for API
                consistency and future migration when ServiceRegistry supports
                the required access patterns.
            handler_type: Specific handler type to retrieve. If None, returns default.

        Returns:
            The registered handler, or None if not found.

        Example:
            >>> handler = RegistryInfraRegistrationStorage.get_handler(
            ...     container,
            ...     handler_type="postgresql",
            ... )
        """
        # Note: Uses module-level storage since ServiceRegistry v1.0 doesn't
        # support dict-style access for multi-handler routing
        if handler_type is not None:
            handler_key = (
                f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.{handler_type}"
            )
        else:
            handler_key = RegistryInfraRegistrationStorage.PROTOCOL_KEY + ".default"

        result = _HANDLER_STORAGE.get(handler_key)
        return cast("ProtocolRegistrationPersistence | None", result)
