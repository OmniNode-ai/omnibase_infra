# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for Intent Storage Node Dependencies.

This module provides RegistryInfraIntentStorage, which registers
dependencies for the NodeIntentStorageEffect node.

Architecture:
    The registry follows ONEX container-based dependency injection:
    - Registers HandlerIntent with ModelONEXContainer
    - Supports initialization with pre-configured HandlerGraph
    - Enables runtime handler resolution via container

    Registration is typically called during application bootstrap.

Related:
    - NodeIntentStorageEffect: Effect node that uses these dependencies
    - HandlerIntent: Intent handler for graph operations
    - HandlerGraph: Underlying Memgraph graph handler
    - ModelONEXContainer: ONEX dependency injection container

Testing:
    This module uses module-level state (``_HANDLER_STORAGE``, ``_PROTOCOL_METADATA``)
    for handler storage. Tests MUST call ``RegistryInfraIntentStorage.clear()`` in
    setup and teardown fixtures to prevent test pollution between test cases.

    Failure to clear state can cause:
    - Tests passing in isolation but failing when run together
    - Handlers from previous tests leaking into subsequent tests
    - Flaky test behavior that is difficult to debug

    Recommended fixture pattern:

    .. code-block:: python

        @pytest.fixture(autouse=True)
        def clear_registry():
            RegistryInfraIntentStorage.clear()
            yield
            RegistryInfraIntentStorage.clear()

Note:
    This registry uses a module-level dict for handler storage because the
    ServiceRegistry in omnibase_core v1.0 doesn't support dict-style access
    or string-keyed multi-handler routing. The handlers are still validated
    but stored separately.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.handlers.handler_intent import HandlerIntent

__all__ = ["RegistryInfraIntentStorage"]

# TODO(ServiceRegistry-v2): Migrate to container.service_registry once v2.0
# supports dict-style access for multi-handler routing. Current module-level
# storage is a workaround for v1.0 limitations. See docstring Note section.
#
# Module-level storage for handlers and metadata
# ServiceRegistry in v1.0 doesn't support dict-style access needed for
# handler routing
_HANDLER_STORAGE: dict[str, object] = {}
_PROTOCOL_METADATA: dict[str, dict[str, object]] = {}


class RegistryInfraIntentStorage:
    """Registry for intent storage node dependencies.

    Registers handler implementations with the ONEX container.
    Supports the HandlerIntent which wraps HandlerGraph for Memgraph operations.

    Usage:
        .. code-block:: python

            from omnibase_core.models.container import ModelONEXContainer
            from omnibase_infra.nodes.node_intent_storage_effect.registry import (
                RegistryInfraIntentStorage,
            )

            # Create container
            container = ModelONEXContainer()

            # Register dependencies
            RegistryInfraIntentStorage.register(container)

            # Register the handler (must have graph_handler configured)
            RegistryInfraIntentStorage.register_handler(
                container,
                handler=intent_handler,
            )

    Note:
        This registry does NOT instantiate handlers. Handlers must be
        created externally with their specific dependencies (HandlerGraph)
        and then registered via register_handler().
    """

    # Handler key for container registration
    HANDLER_KEY = "handler_intent"

    # Default handler type
    DEFAULT_HANDLER_TYPE = "memgraph"

    @staticmethod
    def _is_registered(handler_key: str) -> bool:
        """Check if a handler is already registered for the given key.

        This is a private helper method used to detect re-registration
        attempts, which may indicate container lifecycle issues or
        missing clear() calls in tests.

        Args:
            handler_key: The handler key to check (e.g., "handler_intent.memgraph").

        Returns:
            True if a handler is already registered for this key, False otherwise.
        """
        return handler_key in _HANDLER_STORAGE

    @staticmethod
    def register(_container: ModelONEXContainer) -> None:
        """Register intent storage dependencies with the container.

        Registers the handler key for later handler binding. This method
        sets up the infrastructure but does not bind a specific handler.

        Args:
            _container: ONEX dependency injection container. Currently unused
                because ServiceRegistry v1.0 doesn't support dict-style access
                for multi-handler routing. The parameter is retained for API
                consistency with other registry methods and future migration.

        Example:
            >>> from omnibase_core.models.container import ModelONEXContainer
            >>> container = ModelONEXContainer()
            >>> RegistryInfraIntentStorage.register(container)
        """
        # Register handler metadata for discovery
        # Actual handler binding happens via register_handler()
        _PROTOCOL_METADATA[RegistryInfraIntentStorage.HANDLER_KEY] = {
            "handler": "HandlerIntent",
            "module": "omnibase_infra.handlers.handler_intent",
            "description": "Handler for intent storage operations in Memgraph",
            "capabilities": [
                "intent.storage",
                "intent.storage.store",
                "intent.storage.query_session",
                "intent.storage.query_distribution",
            ],
        }

    @staticmethod
    def register_handler(
        _container: ModelONEXContainer,
        handler: HandlerIntent,
    ) -> None:
        """Register a specific intent handler with the container.

        Binds a concrete HandlerIntent implementation to the handler key.
        The handler must already be initialized with a HandlerGraph.

        Args:
            _container: ONEX dependency injection container. Currently unused
                because ServiceRegistry v1.0 doesn't support dict-style access.
                The parameter is retained for API consistency.
            handler: HandlerIntent instance to register (must be initialized).

        Raises:
            ProtocolConfigurationError: If handler does not implement the required
                protocol methods (initialize, shutdown, execute).

        Example:
            >>> from omnibase_infra.handlers.handler_intent import HandlerIntent
            >>> handler = HandlerIntent(container)
            >>> await handler.initialize({"graph_handler": graph_handler})
            >>> RegistryInfraIntentStorage.register_handler(container, handler)
        """
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import (
            ModelInfraErrorContext,
            ProtocolConfigurationError,
        )

        # NOTE: Protocol-based duck typing is used here per ONEX conventions.
        # We verify the handler implements required methods rather than checking
        # concrete class inheritance. This enables:
        # 1. Fail-fast validation: Immediately reject invalid handlers
        # 2. Duck typing: Any object with the required methods is accepted
        # 3. Testability: Mock handlers can be injected without subclassing
        required_methods = ["initialize", "shutdown", "execute"]
        missing = [
            m for m in required_methods if not callable(getattr(handler, m, None))
        ]
        if missing:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="register_handler",
                target_name="intent_handler",
            )
            raise ProtocolConfigurationError(
                f"Handler missing required protocol methods: {missing}. "
                f"Got {type(handler).__name__}",
                context=context,
            )

        # Store handler in module-level storage
        handler_key = (
            f"{RegistryInfraIntentStorage.HANDLER_KEY}."
            f"{RegistryInfraIntentStorage.DEFAULT_HANDLER_TYPE}"
        )

        # Warn if re-registering over an existing handler
        if RegistryInfraIntentStorage._is_registered(handler_key):
            logger.warning(
                "Re-registering handler '%s'. This may indicate container lifecycle "
                "issues or missing clear() calls in tests.",
                handler_key,
            )

        _HANDLER_STORAGE[handler_key] = handler

        # Also register as default
        default_key = RegistryInfraIntentStorage.HANDLER_KEY + ".default"
        if RegistryInfraIntentStorage._is_registered(default_key):
            logger.warning(
                "Re-registering handler '%s'. This may indicate container lifecycle "
                "issues or missing clear() calls in tests.",
                default_key,
            )
        _HANDLER_STORAGE[default_key] = handler

    @staticmethod
    def get_handler(
        _container: ModelONEXContainer,
        handler_type: str | None = None,
    ) -> HandlerIntent | None:
        """Retrieve a registered intent handler from the container.

        Args:
            _container: ONEX dependency injection container. Currently unused
                because ServiceRegistry v1.0 doesn't support dict-style access.
                The parameter is retained for API consistency.
            handler_type: Specific handler type to retrieve. If None, returns default.

        Returns:
            The registered HandlerIntent, or None if not found.

        Example:
            >>> handler = RegistryInfraIntentStorage.get_handler(container)
            >>> if handler:
            ...     result = await handler.execute(envelope)
        """
        if handler_type is not None:
            handler_key = f"{RegistryInfraIntentStorage.HANDLER_KEY}.{handler_type}"
        else:
            handler_key = RegistryInfraIntentStorage.HANDLER_KEY + ".default"

        result = _HANDLER_STORAGE.get(handler_key)
        return cast("HandlerIntent | None", result)

    @staticmethod
    def clear() -> None:
        """Clear all registered handlers and protocol metadata.

        Resets all module-level state to empty dicts. This method is essential
        for test isolation.

        Warning:
            This method MUST be called in test setup and teardown to prevent
            test pollution. Module-level state persists across test cases within
            the same Python process. Failing to call this method can cause:

            - Handlers from previous tests affecting subsequent tests
            - Tests passing individually but failing when run together
            - Non-deterministic test failures that are difficult to reproduce

        Example:
            .. code-block:: python

                @pytest.fixture(autouse=True)
                def clear_registry():
                    RegistryInfraIntentStorage.clear()
                    yield
                    RegistryInfraIntentStorage.clear()
        """
        _HANDLER_STORAGE.clear()
        _PROTOCOL_METADATA.clear()
