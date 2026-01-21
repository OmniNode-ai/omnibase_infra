# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract-Based Handler Discovery Service for ONEX Infrastructure.

This module provides ContractHandlerDiscovery, which bridges the HandlerPluginLoader
(which loads handler metadata from contracts) with the RegistryProtocolBinding
(which tracks handler classes for runtime instantiation).

Part of OMN-1133: Handler Discovery Service implementation.

The discovery service is responsible for:
- Scanning contract paths for handler definitions
- Loading handler classes via the plugin loader
- Registering discovered handlers with the handler registry
- Tracking errors and warnings without failing the entire operation

Thread Safety:
    The discovery service is stateless and can be safely invoked from multiple
    threads concurrently. All shared state (if any) is delegated to the injected
    plugin_loader and handler_registry, which maintain their own thread safety.

Error Handling Strategy:
    Individual handler failures do NOT fail the entire discovery operation.
    Instead, errors are captured in the ModelDiscoveryResult for the caller
    to handle. This enables graceful degradation where some handlers can be
    registered even if others fail to load.

Example Usage:
    ```python
    from pathlib import Path
    from omnibase_infra.runtime import (
        ContractHandlerDiscovery,
        HandlerPluginLoader,
        get_handler_registry,
    )

    discovery = ContractHandlerDiscovery(
        plugin_loader=HandlerPluginLoader(),
        handler_registry=get_handler_registry(),
    )
    result = await discovery.discover_and_register([Path("nodes/")])
    if result:
        print(f"Registered {result.handlers_registered} handlers")
    else:
        for error in result.errors:
            print(f"Error: {error.message}")
    ```

See Also:
    - HandlerPluginLoader: Loader for reading handler contracts from filesystem
    - RegistryProtocolBinding: Registry for storing discovered handler classes
    - ModelDiscoveryResult: Model representing discovery operation results
    - ProtocolHandlerDiscovery: Protocol interface this class implements

.. versionadded:: 0.7.0
    Created as part of OMN-1133 contract-based handler discovery.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.runtime.model_discovery_error import ModelDiscoveryError
from omnibase_infra.models.runtime.model_discovery_result import ModelDiscoveryResult
from omnibase_infra.models.runtime.model_discovery_warning import (
    ModelDiscoveryWarning,
)
from omnibase_infra.runtime.registry.registry_protocol_binding import RegistryError

if TYPE_CHECKING:
    from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding
    from omnibase_infra.runtime.protocol_handler_plugin_loader import (
        ProtocolHandlerPluginLoader,
    )

logger = logging.getLogger(__name__)


class ContractHandlerDiscovery:
    """Discovers and registers handlers from contracts automatically.

    This class bridges the HandlerPluginLoader (which loads handler metadata from
    contracts) with the RegistryProtocolBinding (which tracks handler classes for
    runtime instantiation).

    Pattern: Stateless service that coordinates between plugin loader and registry.

    The discovery process:
    1. Scans each provided path for handler contracts
    2. Uses the plugin_loader to load and validate handlers
    3. Imports handler classes for registration
    4. Registers classes with the handler_registry
    5. Tracks errors/warnings without failing the entire operation

    Attributes:
        _plugin_loader: Loader for reading handler contracts from filesystem.
        _handler_registry: Registry for storing discovered handler classes.
        _last_discovery_result: Cached result from the most recent discovery operation.
            Populated after each call to ``discover_and_register()``. Enables
            observability tools to query what was discovered without re-running
            discovery. Initially None until first discovery completes.

    Example:
        >>> from pathlib import Path
        >>> discovery = ContractHandlerDiscovery(
        ...     plugin_loader=HandlerPluginLoader(),
        ...     handler_registry=get_handler_registry(),
        ... )
        >>> result = await discovery.discover_and_register([Path("nodes/")])
        >>> if result:
        ...     print(f"Registered {result.handlers_registered} handlers")
        Registered 5 handlers

    .. versionadded:: 0.7.0
    """

    def __init__(
        self,
        plugin_loader: ProtocolHandlerPluginLoader,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Initialize the discovery service.

        Args:
            plugin_loader: Loader for reading handler contracts from filesystem.
                Must implement ProtocolHandlerPluginLoader protocol.
            handler_registry: Registry for storing discovered handler classes.
                Must implement RegistryProtocolBinding interface.
        """
        self._plugin_loader = plugin_loader
        self._handler_registry = handler_registry
        self._last_discovery_result: ModelDiscoveryResult | None = None

    @property
    def last_discovery_result(self) -> ModelDiscoveryResult | None:
        """Return cached discovery result for observability.

        This property provides access to the result of the most recent
        ``discover_and_register()`` call. Useful for observability tools,
        monitoring dashboards, and debugging without re-running discovery.

        Returns:
            The cached ModelDiscoveryResult from the last discovery operation,
            or None if discovery has not been run yet.

        Example:
            >>> discovery = ContractHandlerDiscovery(loader, registry)
            >>> # Before any discovery
            >>> discovery.last_discovery_result is None
            True
            >>> # After discovery
            >>> await discovery.discover_and_register([Path("nodes/")])
            >>> discovery.last_discovery_result.handlers_registered
            5

        .. versionadded:: 0.7.0
        """
        return self._last_discovery_result

    async def discover_and_register(
        self,
        contract_paths: list[Path],
        correlation_id: UUID | None = None,
    ) -> ModelDiscoveryResult:
        """Discover handlers from contracts and register them.

        This is the primary entry point for contract-driven handler discovery.
        It combines path scanning, handler loading, class importing, and
        registration into a single operation.

        Discovery process:
        1. Scan each path for handler contracts (handler_contract.yaml or contract.yaml)
        2. Load and validate handler metadata via plugin_loader
        3. Import handler classes from their module paths
        4. Register classes with handler_registry using handler_name as key
        5. Track errors/warnings without failing entire operation

        Args:
            contract_paths: Paths to scan for handler contracts. Can be:
                - **Directories**: Recursively scanned for contract files
                - **Files**: Directly loaded as contract files
                - **Mixed**: Both directories and files in the same call
            correlation_id: Optional correlation ID for tracing and error context.
                If not provided, a new UUID4 is auto-generated to ensure all
                operations are traceable via structured logging.

        Returns:
            ModelDiscoveryResult containing:
                - ``handlers_discovered``: Number of handlers found
                - ``handlers_registered``: Number successfully registered
                - ``errors``: List of errors (individual failures don't fail operation)
                - ``warnings``: List of non-fatal warnings
                - ``discovered_at``: Timestamp of completion

            The result's ``__bool__`` returns True only if no errors occurred,
            enabling idiomatic success checking: ``if result: ...``

        Raises:
            This method does NOT raise exceptions for individual handler failures.
            All errors are captured in the result's ``errors`` list. Only critical
            system failures (out of memory, etc.) would propagate exceptions.

        Example:
            .. code-block:: python

                from pathlib import Path

                discovery = ContractHandlerDiscovery(
                    plugin_loader=HandlerPluginLoader(),
                    handler_registry=get_handler_registry(),
                )

                # Discover from multiple paths
                result = await discovery.discover_and_register([
                    Path("src/nodes/handlers"),
                    Path("plugins/custom_handler/contract.yaml"),
                ])

                # Check for success
                if result:
                    print(f"Registered {result.handlers_registered} handlers")
                else:
                    for error in result.errors:
                        print(f"Error [{error.error_code}]: {error.message}")

        Thread Safety:
            This method is stateless and can be called concurrently from multiple
            threads or coroutines. The underlying plugin_loader and handler_registry
            are responsible for their own thread safety.

        Note:
            This method performs synchronous file I/O and module imports despite
            being declared async. This is intentional for simplicity since discovery
            is typically performed at application startup when blocking is acceptable.
            For high-concurrency scenarios where blocking the event loop is problematic,
            consider wrapping this call with ``asyncio.to_thread()`` or running it
            in a dedicated thread pool executor.

        .. versionadded:: 0.7.0
        """
        # Auto-generate correlation_id if not provided (per ONEX guidelines)
        correlation_id = correlation_id or uuid4()

        errors: list[ModelDiscoveryError] = []
        warnings: list[ModelDiscoveryWarning] = []
        handlers_discovered = 0
        handlers_registered = 0

        logger.info(
            "Starting handler discovery",
            extra={
                "correlation_id": str(correlation_id),
                "contract_paths": [str(p) for p in contract_paths],
                "path_count": len(contract_paths),
            },
        )

        for path in contract_paths:
            path_handlers_discovered = 0
            path_handlers_registered = 0

            try:
                # Determine path type and load accordingly
                # path.is_dir() and path.is_file() can raise OSError for:
                # - Permission denied when accessing the path
                # - Filesystem errors (unmounted volumes, network failures)
                #
                # Initialize before try block to ensure these variables are always
                # defined, even if an unexpected (non-OSError) exception occurs.
                # This prevents NameError in the outer exception handlers that
                # reference is_file (lines 430, 448, 460).
                is_directory = False
                is_file = False
                try:
                    is_directory = path.is_dir()
                    is_file = path.is_file()
                    path_exists = path.exists()
                except OSError as e:
                    errors.append(
                        ModelDiscoveryError(
                            error_code="PATH_ACCESS_ERROR",
                            message=f"Failed to access path: {e}",
                            contract_path=path,
                            details={"exception_type": type(e).__name__},
                        )
                    )
                    continue

                if not path_exists:
                    errors.append(
                        ModelDiscoveryError(
                            error_code="PATH_NOT_FOUND",
                            message=f"Contract path does not exist: {path}",
                            contract_path=path,
                        )
                    )
                    continue

                # Load handlers based on path type
                if is_directory:
                    # Directory: use load_from_directory for recursive scanning
                    loaded_handlers = self._plugin_loader.load_from_directory(
                        directory=path,
                        correlation_id=correlation_id,
                    )
                elif is_file:
                    # File: use load_from_contract for direct loading
                    loaded_handler = self._plugin_loader.load_from_contract(
                        contract_path=path,
                        correlation_id=correlation_id,
                    )
                    loaded_handlers = [loaded_handler]
                else:
                    # Path exists but is neither file nor directory (e.g., symlink to nothing)
                    errors.append(
                        ModelDiscoveryError(
                            error_code="INVALID_PATH_TYPE",
                            message=f"Path exists but is not a file or directory: {path}",
                            contract_path=path,
                        )
                    )
                    continue

                path_handlers_discovered = len(loaded_handlers)
                handlers_discovered += path_handlers_discovered

                # Register each discovered handler
                for loaded in loaded_handlers:
                    try:
                        # Import the handler class from its fully-qualified path
                        # The plugin_loader already validated the class during loading,
                        # but ModelLoadedHandler only stores the string path
                        handler_class = self._import_handler_class(
                            loaded.handler_class,
                            correlation_id,
                        )

                        # Register with handler_registry using protocol_type as key
                        # This enables runtime lookup by protocol type (e.g., "db", "http")
                        # which matches how message routing selects handlers
                        self._handler_registry.register(
                            loaded.protocol_type,
                            handler_class,
                        )
                        handlers_registered += 1
                        path_handlers_registered += 1

                        logger.debug(
                            "Registered handler: %s (protocol=%s) -> %s",
                            loaded.handler_name,
                            loaded.protocol_type,
                            loaded.handler_class,
                            extra={
                                "correlation_id": str(correlation_id),
                                "handler_name": loaded.handler_name,
                                "protocol_type": loaded.protocol_type,
                                "handler_class": loaded.handler_class,
                                "contract_path": str(loaded.contract_path),
                            },
                        )

                    except RegistryError as e:
                        # Registration failed (e.g., invalid handler protocol)
                        errors.append(
                            ModelDiscoveryError(
                                error_code="REGISTRATION_FAILED",
                                message=f"Failed to register handler: {e}",
                                contract_path=loaded.contract_path,
                                handler_name=loaded.handler_name,
                                details={
                                    "exception_type": type(e).__name__,
                                    "handler_class": loaded.handler_class,
                                },
                            )
                        )
                    except (
                        AttributeError,
                        ImportError,
                        ProtocolConfigurationError,
                    ) as e:
                        # Module/class import failed
                        # AttributeError: Class not found in module (from getattr)
                        # ImportError: Module not found or import failure
                        # ProtocolConfigurationError: Invalid class path or non-class type
                        # Uses IMPORT_ERROR for consistency with EnumHandlerLoaderError
                        errors.append(
                            ModelDiscoveryError(
                                error_code="IMPORT_ERROR",
                                message=f"Failed to import handler class: {e}",
                                contract_path=loaded.contract_path,
                                handler_name=loaded.handler_name,
                                details={
                                    "exception_type": type(e).__name__,
                                    "handler_class": loaded.handler_class,
                                },
                            )
                        )
                    except Exception as e:
                        # CATCH-ALL: Individual handler failures must NOT crash the entire
                        # discovery operation. This enables graceful degradation where some
                        # handlers can be registered even if others fail unexpectedly. All
                        # errors are captured in the result for the caller to handle.
                        errors.append(
                            ModelDiscoveryError(
                                error_code="REGISTRATION_UNEXPECTED_ERROR",
                                message=f"Unexpected error during handler registration: {e}",
                                contract_path=loaded.contract_path,
                                handler_name=loaded.handler_name,
                                details={
                                    "exception_type": type(e).__name__,
                                    "handler_class": loaded.handler_class,
                                },
                            )
                        )

                logger.debug(
                    "Processed path: %s",
                    path,
                    extra={
                        "correlation_id": str(correlation_id),
                        "path": str(path),
                        "handlers_discovered": path_handlers_discovered,
                        "handlers_registered": path_handlers_registered,
                    },
                )

            except ProtocolConfigurationError as e:
                # Plugin loader configuration errors (e.g., invalid contract)
                error_code = "LOAD_ERROR"
                # Extract error code from exception if available
                if hasattr(e, "model") and hasattr(e.model, "context"):
                    context_dict = e.model.context
                    if isinstance(context_dict, dict):
                        loader_error = context_dict.get("loader_error")
                        error_code = (
                            str(loader_error)
                            if loader_error is not None
                            else error_code
                        )

                errors.append(
                    ModelDiscoveryError(
                        error_code=error_code,
                        message=str(e),
                        contract_path=path if is_file else None,
                        details={"exception_type": type(e).__name__},
                    )
                )

            except Exception as e:
                # CATCH-ALL: Discovery must be resilient - a single failing path must NOT
                # crash the entire discovery operation. This enables processing remaining
                # paths even if one path encounters unexpected errors (e.g., filesystem
                # race conditions, corrupted files, unexpected plugin loader exceptions).
                # All errors are captured in the result for the caller to handle.
                #
                # NOTE: Use stored is_file boolean, NOT path.is_file() call
                # which could raise OSError while already handling an exception.
                errors.append(
                    ModelDiscoveryError(
                        error_code="UNEXPECTED_ERROR",
                        message=f"Unexpected error during discovery: {e}",
                        contract_path=path if is_file else None,
                        details={"exception_type": type(e).__name__},
                    )
                )

        result = ModelDiscoveryResult(
            handlers_discovered=handlers_discovered,
            handlers_registered=handlers_registered,
            errors=errors,
            warnings=warnings,
        )

        # Cache result for observability (enables monitoring/debugging without re-running)
        self._last_discovery_result = result

        # Log at appropriate level based on error count
        log_level = logging.WARNING if result.has_errors else logging.INFO
        logger.log(
            log_level,
            "Handler discovery completed: %d discovered, %d registered, %d errors",
            handlers_discovered,
            handlers_registered,
            len(errors),
            extra={
                "correlation_id": str(correlation_id),
                "handlers_discovered": handlers_discovered,
                "handlers_registered": handlers_registered,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "success": not result.has_errors,
            },
        )

        return result

    def _import_handler_class(
        self,
        class_path: str,
        correlation_id: UUID,
    ) -> type:
        """Import handler class from fully qualified path.

        This method imports the handler class specified by the fully-qualified
        class path. Since the HandlerPluginLoader already validated the class
        during loading (checking that it implements ProtocolHandler), re-importing
        here is essentially free due to Python's module caching in sys.modules.

        Args:
            class_path: Fully qualified class path (e.g., 'myapp.handlers.AuthHandler').
                Must contain at least one dot separating module and class name.
            correlation_id: Correlation ID for logging context.

        Returns:
            The imported handler class type.

        Raises:
            ImportError: If module cannot be imported (ModuleNotFoundError).
            AttributeError: If class is not found in the imported module.
            ValueError: If class_path doesn't contain a module separator.

        Example:
            >>> handler_class = discovery._import_handler_class(
            ...     "myapp.handlers.AuthHandler",
            ...     correlation_id=uuid4(),
            ... )
            >>> handler_class.__name__
            'AuthHandler'

        Note:
            This is a synchronous operation despite being called from an async
            method. Python's import system is inherently synchronous, and the
            importlib.import_module() call will block. In practice, this is
            very fast due to module caching.
        """
        if "." not in class_path:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Invalid class path '{class_path}': must be fully qualified "
                "(e.g., 'myapp.handlers.AuthHandler')",
                context=context,
            )

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        handler_class = getattr(module, class_name)

        # Verify it's actually a class (also serves as type narrowing for mypy)
        if not isinstance(handler_class, type):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"'{class_path}' is not a class (got {type(handler_class).__name__})",
                context=context,
            )

        logger.debug(
            "Imported handler class: %s",
            class_path,
            extra={
                "correlation_id": str(correlation_id),
                "class_path": class_path,
                "module_path": module_path,
                "class_name": class_name,
            },
        )

        return handler_class


__all__: list[str] = [
    "ContractHandlerDiscovery",
]
