# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol extension for Infrastructure Handlers with Container DI.

This module defines the ProtocolContainerAware interface, which extends the base
ProtocolHandler from omnibase_spi to add container-based dependency injection
requirements. All infrastructure handlers in omnibase_infra must implement this
extended protocol.

Why This Protocol Exists:
    The base ProtocolHandler in omnibase_spi is intentionally minimal and doesn't
    mandate a specific constructor signature. This keeps the SPI layer decoupled
    from implementation details.

    However, omnibase_infra handlers require ModelONEXContainer for:
    - Dependency injection of shared services (connection pools, clients)
    - Configuration access without global state
    - Testability through container mocking

    This protocol adds the __init__ signature requirement while inheriting all
    method requirements from the base protocol.

Protocol Hierarchy:
    omnibase_spi.ProtocolHandler (base)
        - handler_type property
        - initialize()
        - shutdown()
        - execute()
        - describe()
        - health_check()

    omnibase_infra.ProtocolContainerAware (extension)
        - All methods from ProtocolHandler
        - __init__(container: ModelONEXContainer) requirement

Usage:
    Infrastructure handlers should implement this extended protocol:

    ```python
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.protocols import ProtocolContainerAware

    class HandlerDatabase:
        '''Database handler implementing the infra protocol.'''

        def __init__(self, container: ModelONEXContainer) -> None:
            self._container = container
            # Access shared resources via container

        @property
        def handler_type(self) -> str:
            return "db"

        # ... implement remaining protocol methods ...
    ```

    The RuntimeHostProcess uses this protocol for type-safe handler instantiation:

    ```python
    handler_cls: type[ProtocolContainerAware] = handler_registry.get(handler_type)
    handler_instance = handler_cls(container=container)  # Type-safe
    ```

Thread Safety:
    Handler implementations must be thread-safe. The container provides access
    to shared resources that may be used concurrently.

See Also:
    - omnibase_spi.protocols.handlers.protocol_handler.ProtocolHandler
    - omnibase_core.container.ModelONEXContainer
    - CLAUDE.md section "Container-Based Dependency Injection"
    - PR #186: Container DI refactoring for handlers

.. versionadded:: 0.7.1
    Created as part of OMN-1434 container DI standardization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from omnibase_spi.protocols.handlers.protocol_handler import (
    ProtocolHandler as ProtocolHandlerBase,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

__all__ = [
    "ProtocolContainerAware",
]


@runtime_checkable
class ProtocolContainerAware(ProtocolHandlerBase, Protocol):
    """Extended protocol for infrastructure handlers with container DI.

    This protocol extends the base ProtocolHandler from omnibase_spi to require
    a constructor that accepts ModelONEXContainer for dependency injection.

    All infrastructure handlers in omnibase_infra must implement this protocol.
    The container provides access to:
    - Database connection pools
    - HTTP clients
    - Service discovery clients
    - Configuration values
    - Logging context

    Methods inherited from ProtocolHandler:
        handler_type: Property returning the handler type identifier.
        initialize: Initialize clients and connection pools.
        shutdown: Release resources and close connections.
        execute: Execute protocol-specific operations.
        describe: Return handler metadata and capabilities.
        health_check: Check handler health and connectivity.

    Constructor Requirement (added by this protocol):
        __init__: Must accept container: ModelONEXContainer as first positional argument.

    Example:
        ```python
        from omnibase_core.container import ModelONEXContainer

        class HandlerConsul:
            def __init__(self, container: ModelONEXContainer) -> None:
                self._container = container
                self._consul_url = container.config.get("consul_url")

            @property
            def handler_type(self) -> str:
                return "consul"

            async def initialize(self, config):
                # Use container for shared resources
                ...

            # ... implement remaining methods ...
        ```

    Protocol Verification:
        Per ONEX conventions, verify protocol compliance via duck typing:

        ```python
        handler_cls = handler_registry.get("http")

        # Verify constructor signature accepts container
        import inspect
        sig = inspect.signature(handler_cls.__init__)
        params = list(sig.parameters.keys())
        assert "container" in params or len(params) > 1  # self + container

        # Verify protocol methods exist
        assert hasattr(handler_cls, "handler_type")
        assert hasattr(handler_cls, "initialize") and callable(getattr(handler_cls, "initialize"))
        ```

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the handler with a dependency injection container.

        All infrastructure handlers receive their dependencies through the
        container rather than as individual constructor arguments. This enables:

        - Consistent initialization across all handlers
        - Easy testing through container mocking
        - Runtime configuration without code changes
        - Shared resource management (connection pools, clients)

        Args:
            container: ONEX dependency injection container providing access to:
                - Configuration values (container.config)
                - Shared services (database pools, HTTP clients)
                - Logging context
                - Runtime metadata

        Note:
            Handlers should store the container reference and access dependencies
            lazily when needed, rather than extracting all values in __init__.
            This improves startup time and allows for late-bound configuration.

        Example:
            ```python
            def __init__(self, container: ModelONEXContainer) -> None:
                self._container = container
                self._client: httpx.AsyncClient | None = None  # Lazy init

            async def initialize(self, config):
                # Create client during initialize(), not __init__()
                self._client = httpx.AsyncClient(
                    base_url=self._container.config.get("base_url"),
                    timeout=config.get("timeout", 30.0),
                )
            ```
        """
        ...
