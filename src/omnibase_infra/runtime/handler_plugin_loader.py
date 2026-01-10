# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Plugin Loader for Contract-Driven Discovery.

This module provides HandlerPluginLoader, which discovers handler contracts
from the filesystem, validates handlers against protocols, and creates
ModelLoadedHandler instances for runtime registration.

Part of OMN-1132: Handler Plugin Loader implementation.

The loader implements ProtocolHandlerPluginLoader and supports:
- Single contract loading from a specific path
- Directory-based discovery with recursive scanning
- Glob pattern-based discovery for flexible matching

Thread Safety:
    The loader is designed to be stateless and safe for concurrent use
    from multiple threads. Each load operation is independent:

    - No instance state is stored after ``__init__`` (empty constructor)
    - All method variables are local to each call (thread-local by nature)
    - ``importlib.import_module()`` is thread-safe in CPython (uses import lock)
    - File operations use independent file handles per call

    Caveat: The ``discover_and_load()`` method uses ``Path.cwd()`` by default,
    which reads process-level state. For deterministic behavior in multi-threaded
    environments, provide an explicit ``base_path`` parameter.

See Also:
    - ProtocolHandlerPluginLoader: Protocol definition for plugin loaders
    - HandlerContractSource: Contract discovery and parsing
    - ModelLoadedHandler: Model representing loaded handler metadata

Security Considerations:
    This loader dynamically imports Python classes specified in YAML contracts.
    Contract files should be treated as code and protected accordingly:
    - Only load contracts from trusted sources
    - Validate contract file permissions in production environments
    - Be aware that module side effects execute during import
    - Consider allowlisting import paths in high-security environments

.. versionadded:: 0.7.0
    Created as part of OMN-1132 handler plugin loader implementation.
"""

from __future__ import annotations

import importlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import yaml
from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerLoaderError, EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ProtocolConfigurationError
from omnibase_infra.models.errors import ModelInfraErrorContext
from omnibase_infra.models.runtime import (
    ModelFailedPluginLoad,
    ModelHandlerContract,
    ModelLoadedHandler,
    ModelPluginLoadContext,
    ModelPluginLoadSummary,
)
from omnibase_infra.runtime.protocol_handler_plugin_loader import (
    ProtocolHandlerPluginLoader,
)

if TYPE_CHECKING:
    from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler

logger = logging.getLogger(__name__)

# File pattern for handler contracts
HANDLER_CONTRACT_FILENAME = "handler_contract.yaml"
CONTRACT_YAML_FILENAME = "contract.yaml"

# Maximum contract file size (10MB) to prevent memory exhaustion
MAX_CONTRACT_SIZE = 10 * 1024 * 1024


class HandlerPluginLoader(ProtocolHandlerPluginLoader):
    """Load handlers as plugins from contracts.

    Discovers handler contracts, validates handlers against protocols,
    and registers them with the handler registry.

    This class implements ProtocolHandlerPluginLoader by scanning filesystem
    paths for handler_contract.yaml or contract.yaml files, parsing them,
    dynamically importing the handler classes, and creating ModelLoadedHandler
    instances.

    Protocol Compliance:
        This class explicitly implements ProtocolHandlerPluginLoader and provides
        all required methods: load_from_contract(), load_from_directory(), and
        discover_and_load(). Protocol compliance is verified via duck typing.

    Example:
        >>> # Load a single handler from contract
        >>> loader = HandlerPluginLoader()
        >>> handler = loader.load_from_contract(
        ...     Path("src/handlers/auth/handler_contract.yaml")
        ... )
        >>> print(f"Loaded: {handler.handler_name}")

        >>> # Load all handlers from a directory
        >>> handlers = loader.load_from_directory(Path("src/handlers"))
        >>> print(f"Loaded {len(handlers)} handlers")

        >>> # Discover with glob patterns
        >>> handlers = loader.discover_and_load([
        ...     "src/**/handler_contract.yaml",
        ...     "plugins/**/contract.yaml",
        ... ])

    .. versionadded:: 0.7.0
        Created as part of OMN-1132 handler plugin loader implementation.
    """

    def __init__(self) -> None:
        """Initialize the handler plugin loader.

        The loader is stateless and does not require any configuration.
        All operations are performed on-demand based on provided paths.
        """

    def load_from_contract(
        self,
        contract_path: Path,
        correlation_id: str | None = None,
    ) -> ModelLoadedHandler:
        """Load a single handler from a contract file.

        Parses the contract YAML file at the given path, validates it,
        imports the handler class, validates protocol compliance, and
        returns a ModelLoadedHandler with the loaded metadata.

        Args:
            contract_path: Path to the handler contract YAML file.
                Must be an absolute or relative path to an existing file.
            correlation_id: Optional correlation ID for tracing and error context.
                If not provided, a new UUID4 is auto-generated to ensure all
                operations have traceable correlation IDs.

        Returns:
            ModelLoadedHandler containing the loaded handler metadata
            including handler class, version, and contract information.

        Raises:
            ProtocolConfigurationError: If the contract file is invalid,
                missing required fields, or fails validation. Error codes:
                - HANDLER_LOADER_001: Contract file not found (path doesn't exist)
                - HANDLER_LOADER_002: Invalid YAML syntax
                - HANDLER_LOADER_003: Schema validation failed
                - HANDLER_LOADER_004: Missing required fields
                - HANDLER_LOADER_005: Contract file exceeds size limit
                - HANDLER_LOADER_006: Handler does not implement protocol
                - HANDLER_LOADER_007: Path exists but is not a file (e.g., directory)
                - HANDLER_LOADER_008: Failed to read contract file (I/O error)
                - HANDLER_LOADER_009: Failed to stat contract file (I/O error)
            InfraConnectionError: If the handler class cannot be imported.
                Error codes:
                - HANDLER_LOADER_010: Module not found
                - HANDLER_LOADER_011: Class not found in module
                - HANDLER_LOADER_012: Import error (syntax/dependency)
        """
        # Auto-generate correlation_id if not provided (per ONEX guidelines)
        correlation_id = correlation_id or str(uuid4())

        logger.debug(
            "Loading handler from contract: %s",
            contract_path,
            extra={
                "contract_path": str(contract_path),
                "correlation_id": correlation_id,
            },
        )

        # Validate contract path exists
        if not contract_path.exists():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Contract file not found: {contract_path}",
                context=context,
                loader_error=EnumHandlerLoaderError.FILE_NOT_FOUND.value,
                contract_path=str(contract_path),
            )

        if not contract_path.is_file():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Contract path is not a file: {contract_path}",
                context=context,
                loader_error=EnumHandlerLoaderError.NOT_A_FILE.value,
                contract_path=str(contract_path),
            )

        # Validate file size (raises ProtocolConfigurationError on failure)
        self._validate_file_size(
            contract_path,
            correlation_id=correlation_id,
            operation="load_from_contract",
            raise_on_error=True,
        )

        # Parse YAML contract
        try:
            with contract_path.open("r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Invalid YAML syntax in contract: {e}",
                context=context,
                loader_error=EnumHandlerLoaderError.INVALID_YAML_SYNTAX.value,
                contract_path=str(contract_path),
            ) from e
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Failed to read contract file: {e}",
                context=context,
                loader_error=EnumHandlerLoaderError.FILE_READ_ERROR.value,
                contract_path=str(contract_path),
            ) from e

        if raw_data is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Contract file is empty",
                context=context,
                loader_error=EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value,
                contract_path=str(contract_path),
            )

        # Validate contract using Pydantic model
        try:
            contract = ModelHandlerContract.model_validate(raw_data)
        except ValidationError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            # Convert validation errors to readable message
            error_details = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            raise ProtocolConfigurationError(
                f"Contract validation failed: {error_details}",
                context=context,
                loader_error=EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value,
                contract_path=str(contract_path),
                validation_errors=[
                    {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
                    for err in e.errors()
                ],
            ) from e

        handler_name = contract.handler_name
        handler_class_path = contract.handler_class
        handler_type = contract.handler_type
        capability_tags = contract.capability_tags

        # Import and validate handler class
        handler_class = self._import_handler_class(
            handler_class_path, contract_path, correlation_id
        )

        # Validate handler implements protocol
        is_valid, missing_methods = self._validate_handler_protocol(handler_class)
        if not is_valid:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            missing_str = ", ".join(missing_methods)
            raise ProtocolConfigurationError(
                f"Handler class {handler_class_path} does not implement "
                f"ProtocolHandler (missing required method(s): {missing_str})",
                context=context,
                loader_error=EnumHandlerLoaderError.PROTOCOL_NOT_IMPLEMENTED.value,
                contract_path=str(contract_path),
                handler_class=handler_class_path,
                missing_methods=missing_methods,
            )

        logger.info(
            "Successfully loaded handler from contract: %s -> %s",
            handler_name,
            handler_class_path,
            extra={
                "handler_name": handler_name,
                "handler_class": handler_class_path,
                "handler_type": handler_type.value,
                "contract_path": str(contract_path),
            },
        )

        return ModelLoadedHandler(
            handler_name=handler_name,
            handler_type=handler_type,
            handler_class=handler_class_path,
            contract_path=contract_path.resolve(),
            capability_tags=capability_tags,
            loaded_at=datetime.now(UTC),
        )

    def load_from_directory(
        self,
        directory: Path,
        correlation_id: str | None = None,
        max_handlers: int | None = None,
    ) -> list[ModelLoadedHandler]:
        """Load all handlers from contract files in a directory.

        Recursively scans the given directory for handler contract files
        (handler_contract.yaml or contract.yaml), loads each handler,
        and returns a list of successfully loaded handlers.

        Failed loads are logged but do not stop processing of other handlers.
        A summary is logged at the end of the operation for observability.

        Args:
            directory: Path to the directory to scan for contract files.
                Must be an existing directory.
            correlation_id: Optional correlation ID for tracing and error context.
                If not provided, a new UUID4 is auto-generated to ensure all
                operations have traceable correlation IDs. The same correlation_id
                is propagated to all contract loads within the directory scan.
            max_handlers: Optional maximum number of handlers to discover and load.
                If specified, discovery stops after finding this many contract files.
                A warning is logged when the limit is reached. Set to None (default)
                for unlimited discovery. This prevents runaway resource usage when
                scanning directories with unexpectedly large numbers of handlers.

        Returns:
            List of successfully loaded handlers. May be empty if no
            contracts are found or all fail validation.

        Raises:
            ProtocolConfigurationError: If the directory does not exist
                or is not accessible. Error codes:
                - HANDLER_LOADER_020: Directory not found
                - HANDLER_LOADER_021: Permission denied
                - HANDLER_LOADER_022: Not a directory
        """
        # Auto-generate correlation_id if not provided (per ONEX guidelines)
        correlation_id = correlation_id or str(uuid4())

        # Start timing for observability
        start_time = time.perf_counter()

        logger.debug(
            "Loading handlers from directory: %s",
            directory,
            extra={
                "directory": str(directory),
                "correlation_id": correlation_id,
                "max_handlers": max_handlers,
            },
        )

        # Validate directory exists
        if not directory.exists():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_directory",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Directory not found: {directory}",
                context=context,
                loader_error=EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.value,
                directory=str(directory),
            )

        if not directory.is_dir():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_directory",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Path is not a directory: {directory}",
                context=context,
                loader_error=EnumHandlerLoaderError.NOT_A_DIRECTORY.value,
                directory=str(directory),
            )

        # Find all contract files (with optional limit)
        contract_files = self._find_contract_files(
            directory, correlation_id, max_handlers=max_handlers
        )

        logger.debug(
            "Found %d contract files in directory: %s",
            len(contract_files),
            directory,
            extra={
                "directory": str(directory),
                "contract_count": len(contract_files),
            },
        )

        # Load each contract (graceful mode - continue on errors)
        handlers: list[ModelLoadedHandler] = []
        failed_handlers: list[ModelFailedPluginLoad] = []

        for contract_path in contract_files:
            try:
                handler = self.load_from_contract(contract_path, correlation_id)
                handlers.append(handler)
            except (ProtocolConfigurationError, InfraConnectionError) as e:
                # Extract error code if available
                error_code: str | None = None
                if hasattr(e, "model") and hasattr(e.model, "context"):
                    error_code = e.model.context.get("loader_error")

                failed_handlers.append(
                    ModelFailedPluginLoad(
                        contract_path=contract_path,
                        error_message=str(e),
                        error_code=error_code,
                    )
                )

                logger.warning(
                    "Failed to load handler from %s: %s",
                    contract_path,
                    str(e),
                    extra={
                        "contract_path": str(contract_path),
                        "error": str(e),
                        "error_code": error_code,
                        "correlation_id": correlation_id,
                    },
                )
                continue

        # Calculate duration and log summary
        duration_seconds = time.perf_counter() - start_time

        self._log_load_summary(
            ModelPluginLoadContext(
                operation="load_from_directory",
                source=str(directory),
                total_discovered=len(contract_files),
                handlers=handlers,
                failed_plugins=failed_handlers,
                duration_seconds=duration_seconds,
                correlation_id=UUID(correlation_id),
            )
        )

        return handlers

    def discover_and_load(
        self,
        patterns: list[str],
        correlation_id: str | None = None,
        base_path: Path | None = None,
        max_handlers: int | None = None,
    ) -> list[ModelLoadedHandler]:
        """Discover contracts matching glob patterns and load handlers.

        Searches for contract files matching the given glob patterns,
        deduplicates matches, loads each handler, and returns a list
        of successfully loaded handlers.

        A summary is logged at the end of the operation for observability.

        Working Directory Dependency:
            By default, glob patterns are resolved relative to the current
            working directory (``Path.cwd()``). This means results may vary
            if the working directory changes between calls. For deterministic
            behavior in environments where cwd may change (e.g., tests,
            multi-threaded applications), provide an explicit ``base_path``
            parameter.

        Args:
            patterns: List of glob patterns to match contract files.
                Supports standard glob syntax including ** for recursive.
            correlation_id: Optional correlation ID for tracing and error context.
                If not provided, a new UUID4 is auto-generated to ensure all
                operations have traceable correlation IDs. The same correlation_id
                is propagated to all discovered contract loads.
            base_path: Optional base path for resolving glob patterns.
                If not provided, defaults to ``Path.cwd()``. Providing an
                explicit base path ensures deterministic behavior regardless
                of the current working directory.
            max_handlers: Optional maximum number of handlers to discover and load.
                If specified, discovery stops after finding this many contract files.
                A warning is logged when the limit is reached. Set to None (default)
                for unlimited discovery. This prevents runaway resource usage when
                scanning directories with unexpectedly large numbers of handlers.

        Returns:
            List of successfully loaded handlers. May be empty if no
            patterns match or all fail validation.

        Raises:
            ProtocolConfigurationError: If patterns list is empty.
                Error codes:
                - HANDLER_LOADER_030: Empty patterns list

        Example:
            >>> # Using default cwd-based resolution
            >>> handlers = loader.discover_and_load(["src/**/handler_contract.yaml"])
            >>>
            >>> # Using explicit base path for deterministic behavior
            >>> handlers = loader.discover_and_load(
            ...     ["src/**/handler_contract.yaml"],
            ...     base_path=Path("/app/project"),
            ... )
        """
        # Auto-generate correlation_id if not provided (per ONEX guidelines)
        correlation_id = correlation_id or str(uuid4())

        # Start timing for observability
        start_time = time.perf_counter()

        logger.debug(
            "Discovering handlers with patterns: %s",
            patterns,
            extra={
                "patterns": patterns,
                "correlation_id": correlation_id,
                "max_handlers": max_handlers,
            },
        )

        if not patterns:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="discover_and_load",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Patterns list cannot be empty",
                context=context,
                loader_error=EnumHandlerLoaderError.EMPTY_PATTERNS_LIST.value,
            )

        # Collect all matching contract files, deduplicated by resolved path
        discovered_paths: set[Path] = set()
        limit_reached = False

        # Use explicit base_path if provided, otherwise fall back to cwd
        # Note: Using cwd can produce different results if the working directory
        # changes between calls. For deterministic behavior, provide base_path.
        glob_base = base_path if base_path is not None else Path.cwd()

        for pattern in patterns:
            if limit_reached:
                break

            matched_paths = list(glob_base.glob(pattern))
            for path in matched_paths:
                # Check if we've reached the limit
                if max_handlers is not None and len(discovered_paths) >= max_handlers:
                    limit_reached = True
                    logger.warning(
                        "Handler discovery limit reached: stopped after discovering %d "
                        "handlers (max_handlers=%d). Some handlers may not be loaded.",
                        len(discovered_paths),
                        max_handlers,
                        extra={
                            "discovered_count": len(discovered_paths),
                            "max_handlers": max_handlers,
                            "patterns": patterns,
                            "correlation_id": correlation_id,
                        },
                    )
                    break

                if path.is_file():
                    # Early size validation to skip oversized files before expensive operations
                    if (
                        self._validate_file_size(
                            path, correlation_id=correlation_id, raise_on_error=False
                        )
                        is None
                    ):
                        continue

                    # Early YAML syntax validation to fail fast before expensive resolve operations
                    # This catches malformed YAML immediately after discovery rather than
                    # deferring to load_from_contract, which is more efficient for batch discovery
                    if not self._validate_yaml_syntax(
                        path, correlation_id=correlation_id, raise_on_error=False
                    ):
                        continue

                    resolved = path.resolve()
                    discovered_paths.add(resolved)

        logger.debug(
            "Discovered %d unique contract files from %d patterns",
            len(discovered_paths),
            len(patterns),
            extra={
                "patterns": patterns,
                "discovered_count": len(discovered_paths),
                "limit_reached": limit_reached,
            },
        )

        # Load each discovered contract (graceful mode)
        handlers: list[ModelLoadedHandler] = []
        failed_handlers: list[ModelFailedPluginLoad] = []

        for contract_path in sorted(discovered_paths):
            try:
                handler = self.load_from_contract(contract_path, correlation_id)
                handlers.append(handler)
            except (ProtocolConfigurationError, InfraConnectionError) as e:
                # Extract error code if available
                error_code: str | None = None
                if hasattr(e, "model") and hasattr(e.model, "context"):
                    error_code = e.model.context.get("loader_error")

                failed_handlers.append(
                    ModelFailedPluginLoad(
                        contract_path=contract_path,
                        error_message=str(e),
                        error_code=error_code,
                    )
                )

                logger.warning(
                    "Failed to load handler from %s: %s",
                    contract_path,
                    str(e),
                    extra={
                        "contract_path": str(contract_path),
                        "error": str(e),
                        "error_code": error_code,
                        "correlation_id": correlation_id,
                    },
                )
                continue

        # Calculate duration and log summary
        duration_seconds = time.perf_counter() - start_time

        # Format patterns as comma-separated string for source
        patterns_str = ", ".join(patterns)

        self._log_load_summary(
            ModelPluginLoadContext(
                operation="discover_and_load",
                source=patterns_str,
                total_discovered=len(discovered_paths),
                handlers=handlers,
                failed_plugins=failed_handlers,
                duration_seconds=duration_seconds,
                correlation_id=UUID(correlation_id),
            )
        )

        return handlers

    def _log_load_summary(
        self,
        context: ModelPluginLoadContext,
    ) -> ModelPluginLoadSummary:
        """Log a summary of the handler loading operation for observability.

        Creates a structured summary of the load operation and logs it at
        an appropriate level (INFO for success, WARNING if there were failures).

        The log message format is designed for easy parsing:
        - Single line summary with counts and timing
        - Detailed handler list with class names and modules
        - Failed handler details with error reasons

        Args:
            context: The load context containing operation details, handlers,
                failures, and timing information.

        Returns:
            ModelPluginLoadSummary containing the structured summary data.

        Example log output:
            Handler load complete: 5 handlers loaded in 0.23s (source: /app/handlers)
              - HandlerAuth (myapp.handlers.auth)
              - HandlerDb (myapp.handlers.db)
              ...
        """
        # Build list of loaded handler details
        loaded_handler_details = [
            {
                "name": h.handler_name,
                "class": h.handler_class.rsplit(".", 1)[-1],
                "module": h.handler_class.rsplit(".", 1)[0],
            }
            for h in context.handlers
        ]

        # Create summary model
        summary = ModelPluginLoadSummary(
            operation=context.operation,
            source=context.source,
            total_discovered=context.total_discovered,
            total_loaded=len(context.handlers),
            total_failed=len(context.failed_plugins),
            loaded_plugins=loaded_handler_details,
            failed_plugins=context.failed_plugins,
            duration_seconds=context.duration_seconds,
            correlation_id=context.correlation_id,
            completed_at=datetime.now(UTC),
        )

        # Build log message with handler details
        handler_lines = [
            f"  - {h['class']} ({h['module']})" for h in loaded_handler_details
        ]
        handler_list_str = "\n".join(handler_lines) if handler_lines else "  (none)"

        # Build failed handler message if any
        failed_lines = []
        for failed in context.failed_plugins:
            error_code_str = f" [{failed.error_code}]" if failed.error_code else ""
            failed_lines.append(f"  - {failed.contract_path}{error_code_str}")

        failed_list_str = "\n".join(failed_lines) if failed_lines else ""

        # Choose log level based on whether there were failures
        if context.failed_plugins:
            log_level = logging.WARNING
            status = "with failures"
        else:
            log_level = logging.INFO
            status = "successfully"

        # Format duration for readability
        if context.duration_seconds < 0.001:
            duration_str = f"{context.duration_seconds * 1000000:.0f}us"
        elif context.duration_seconds < 1.0:
            duration_str = f"{context.duration_seconds * 1000:.2f}ms"
        else:
            duration_str = f"{context.duration_seconds:.2f}s"

        # Log the summary
        summary_msg = (
            f"Handler load complete {status}: "
            f"{len(context.handlers)} handlers loaded in {duration_str}"
        )
        if context.failed_plugins:
            summary_msg += f" ({len(context.failed_plugins)} failed)"

        # Build detailed message
        detailed_msg = f"{summary_msg}\nLoaded handlers:\n{handler_list_str}"
        if failed_list_str:
            detailed_msg += f"\nFailed handlers:\n{failed_list_str}"

        logger.log(
            log_level,
            detailed_msg,
            extra={
                "operation": context.operation,
                "source": context.source,
                "total_discovered": context.total_discovered,
                "total_loaded": len(context.handlers),
                "total_failed": len(context.failed_plugins),
                "duration_seconds": context.duration_seconds,
                "correlation_id": str(context.correlation_id),
                "handler_names": [h.handler_name for h in context.handlers],
                "handler_classes": [h.handler_class for h in context.handlers],
                "failed_paths": [str(f.contract_path) for f in context.failed_plugins],
            },
        )

        return summary

    def _validate_handler_protocol(self, handler_class: type) -> tuple[bool, list[str]]:
        """Validate handler implements required protocol (ProtocolHandler).

        Uses duck typing to verify the handler class has the required
        methods for ProtocolHandler compliance. Per ONEX conventions, protocol
        compliance is verified via structural typing (duck typing) rather than
        isinstance checks or explicit inheritance.

        Protocol Requirements (from omnibase_spi.protocols.handlers.protocol_handler):
            The ProtocolHandler protocol defines the following required members:

            **Required Methods (validated)**:
                - ``handler_type`` (property): Returns handler type identifier string
                - ``initialize(config)``: Async method to initialize connections/pools
                - ``shutdown(timeout_seconds)``: Async method to release resources
                - ``execute(request, operation_config)``: Async method for operations
                - ``describe()``: Sync method returning handler metadata/capabilities

            **Optional Methods (not validated)**:
                - ``health_check()``: Async method for connectivity verification.
                  While part of the ProtocolHandler protocol, this method is not
                  validated because existing handler implementations (HandlerHttp,
                  HandlerDb, HandlerVault, HandlerConsul) do not implement it.
                  Future handler implementations SHOULD include health_check().

        Validation Approach:
            This method checks for the presence and callability of all 5 required
            methods. A handler class must have ALL of these methods to pass validation.
            This prevents false positives where a class might have only ``describe()``
            but lack other essential handler functionality.

            The validation uses ``callable(getattr(...))`` for methods and
            ``hasattr()`` for the ``handler_type`` property to accommodate both
            instance properties and class-level descriptors.

        Why Duck Typing:
            ONEX uses duck typing for protocol validation to:
            1. Avoid tight coupling to specific base classes
            2. Enable flexibility in handler implementation strategies
            3. Support mixin-based handler composition
            4. Allow testing with mock handlers that satisfy the protocol

        Args:
            handler_class: The handler class to validate. Must be a class type,
                not an instance.

        Returns:
            A tuple of (is_valid, missing_methods) where:
            - is_valid: True if handler implements all required protocol methods
            - missing_methods: List of method names that are missing or not callable.
              Empty list if all methods are present.

        Example:
            >>> class ValidHandler:
            ...     @property
            ...     def handler_type(self) -> str: return "test"
            ...     async def initialize(self, config): pass
            ...     async def shutdown(self, timeout_seconds=30.0): pass
            ...     async def execute(self, request, config): pass
            ...     def describe(self): return {}
            ...
            >>> loader = HandlerPluginLoader()
            >>> loader._validate_handler_protocol(ValidHandler)
            (True, [])

            >>> class IncompleteHandler:
            ...     def describe(self): return {}
            ...
            >>> loader._validate_handler_protocol(IncompleteHandler)
            (False, ['handler_type', 'initialize', 'shutdown', 'execute'])

        See Also:
            - ``omnibase_spi.protocols.handlers.protocol_handler.ProtocolHandler``
            - ``docs/architecture/RUNTIME_HOST_IMPLEMENTATION_PLAN.md``
        """
        # Check for required ProtocolHandler methods via duck typing
        # All 5 core methods must be present for protocol compliance
        missing_methods: list[str] = []

        # 1. handler_type property - can be property or method
        if not hasattr(handler_class, "handler_type"):
            missing_methods.append("handler_type")

        # 2. initialize() - async method for connection setup
        if not callable(getattr(handler_class, "initialize", None)):
            missing_methods.append("initialize")

        # 3. shutdown() - async method for resource cleanup
        if not callable(getattr(handler_class, "shutdown", None)):
            missing_methods.append("shutdown")

        # 4. execute() - async method for operation execution
        if not callable(getattr(handler_class, "execute", None)):
            missing_methods.append("execute")

        # 5. describe() - sync method for introspection
        if not callable(getattr(handler_class, "describe", None)):
            missing_methods.append("describe")

        # Note: health_check() is part of ProtocolHandler but is NOT validated
        # because existing handlers (HandlerHttp, HandlerDb, etc.) do not
        # implement it. Future handlers SHOULD implement health_check().

        return (len(missing_methods) == 0, missing_methods)

    def _import_handler_class(
        self,
        class_path: str,
        contract_path: Path,
        correlation_id: str | None = None,
    ) -> type:
        """Dynamically import handler class from fully qualified path.

        Args:
            class_path: Fully qualified class path (e.g., 'myapp.handlers.AuthHandler').
            contract_path: Path to the contract file (for error context).
            correlation_id: Optional correlation ID for tracing and error context.

        Returns:
            The imported class type.

        Raises:
            InfraConnectionError: If the module or class cannot be imported.
                Error codes include correlation_id when provided for traceability.
        """
        # Split class path into module and class name
        if "." not in class_path:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Invalid class path '{class_path}': must be fully qualified "
                "(e.g., 'myapp.handlers.AuthHandler')",
                context=context,
                loader_error=EnumHandlerLoaderError.MODULE_NOT_FOUND.value,
                class_path=class_path,
                contract_path=str(contract_path),
            )

        module_path, class_name = class_path.rsplit(".", 1)

        # Import the module
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Module not found: {module_path}",
                context=context,
                loader_error=EnumHandlerLoaderError.MODULE_NOT_FOUND.value,
                module_path=module_path,
                class_path=class_path,
                contract_path=str(contract_path),
            ) from e
        except ImportError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Import error loading module {module_path}: {e}",
                context=context,
                loader_error=EnumHandlerLoaderError.IMPORT_ERROR.value,
                module_path=module_path,
                class_path=class_path,
                contract_path=str(contract_path),
            ) from e

        # Get the class from the module
        if not hasattr(module, class_name):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Class '{class_name}' not found in module '{module_path}'",
                context=context,
                loader_error=EnumHandlerLoaderError.CLASS_NOT_FOUND.value,
                module_path=module_path,
                class_name=class_name,
                class_path=class_path,
                contract_path=str(contract_path),
            )

        handler_class = getattr(module, class_name)

        # Verify it's actually a class
        if not isinstance(handler_class, type):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"'{class_path}' is not a class",
                context=context,
                loader_error=EnumHandlerLoaderError.CLASS_NOT_FOUND.value,
                class_path=class_path,
                contract_path=str(contract_path),
            )

        return handler_class

    def _validate_yaml_syntax(
        self,
        path: Path,
        correlation_id: str | None = None,
        raise_on_error: bool = True,
    ) -> bool:
        """Validate YAML syntax of a contract file for early fail-fast behavior.

        Performs early YAML syntax validation to fail fast before expensive
        operations like path resolution and handler class loading. This method
        only validates that the file contains valid YAML syntax; it does not
        perform schema validation.

        This enables the discover_and_load method to skip malformed YAML files
        immediately after discovery, rather than deferring the error to
        load_from_contract which would be less efficient for large discovery
        operations.

        Args:
            path: Path to the YAML file to validate. Must be an existing file.
            correlation_id: Optional correlation ID for error context.
            raise_on_error: If True (default), raises ProtocolConfigurationError
                on YAML syntax errors. If False, logs a warning and returns False,
                allowing the caller to skip the file.

        Returns:
            True if YAML syntax is valid.
            False if raise_on_error is False and YAML syntax is invalid.

        Raises:
            ProtocolConfigurationError: If raise_on_error is True and:
                - INVALID_YAML_SYNTAX: File contains invalid YAML syntax
                - FILE_READ_ERROR: Failed to read file (I/O error)

        Note:
            The error message includes the YAML parser error details which
            typically contain line and column information for the syntax error.
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            if raise_on_error:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="validate_yaml_syntax",
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"Invalid YAML syntax in contract file '{path}': {e}",
                    context=context,
                    loader_error=EnumHandlerLoaderError.INVALID_YAML_SYNTAX.value,
                    contract_path=str(path),
                ) from e
            logger.warning(
                "Skipping contract file with invalid YAML syntax %s: %s",
                path,
                e,
                extra={
                    "path": str(path),
                    "error": str(e),
                    "correlation_id": correlation_id,
                },
            )
            return False
        except OSError as e:
            if raise_on_error:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="validate_yaml_syntax",
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"Failed to read contract file '{path}': {e}",
                    context=context,
                    loader_error=EnumHandlerLoaderError.FILE_READ_ERROR.value,
                    contract_path=str(path),
                ) from e
            logger.warning(
                "Failed to read contract file %s: %s",
                path,
                e,
                extra={
                    "path": str(path),
                    "error": str(e),
                    "correlation_id": correlation_id,
                },
            )
            return False

        return True

    def _validate_file_size(
        self,
        path: Path,
        correlation_id: str | None = None,
        operation: str = "load_from_contract",
        raise_on_error: bool = True,
    ) -> int | None:
        """Validate file size is within limits.

        Checks that the file at the given path can be stat'd and does not
        exceed MAX_CONTRACT_SIZE. Supports both strict mode (raising exceptions)
        and graceful mode (logging warnings and returning None).

        Args:
            path: Path to the file to validate. Must be an existing file.
            correlation_id: Optional correlation ID for error context.
            operation: The operation name for error context in exceptions.
            raise_on_error: If True (default), raises ProtocolConfigurationError
                on stat failure or size exceeded. If False, logs a warning
                and returns None, allowing the caller to skip the file.

        Returns:
            File size in bytes if validation passes.
            None if raise_on_error is False and validation fails (stat error
            or size exceeded).

        Raises:
            ProtocolConfigurationError: If raise_on_error is True and:
                - FILE_STAT_ERROR: Failed to stat the file (I/O error)
                - FILE_SIZE_EXCEEDED: File exceeds MAX_CONTRACT_SIZE
        """
        # Attempt to get file size
        try:
            file_size = path.stat().st_size
        except OSError as e:
            if raise_on_error:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation=operation,
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"Failed to stat contract file: {e}",
                    context=context,
                    loader_error=EnumHandlerLoaderError.FILE_STAT_ERROR.value,
                    contract_path=str(path),
                ) from e
            logger.warning(
                "Failed to stat contract file %s: %s",
                path,
                e,
                extra={"path": str(path), "error": str(e)},
            )
            return None

        # Check size limit
        if file_size > MAX_CONTRACT_SIZE:
            if raise_on_error:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation=operation,
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"Contract file exceeds size limit: {file_size} bytes "
                    f"(max: {MAX_CONTRACT_SIZE} bytes)",
                    context=context,
                    loader_error=EnumHandlerLoaderError.FILE_SIZE_EXCEEDED.value,
                    contract_path=str(path),
                    file_size=file_size,
                    max_size=MAX_CONTRACT_SIZE,
                )
            logger.warning(
                "Skipping oversized contract file %s: %d bytes exceeds limit of %d bytes",
                path,
                file_size,
                MAX_CONTRACT_SIZE,
                extra={
                    "path": str(path),
                    "file_size": file_size,
                    "max_size": MAX_CONTRACT_SIZE,
                },
            )
            return None

        return file_size

    def _find_contract_files(
        self,
        directory: Path,
        correlation_id: str | None = None,
        max_handlers: int | None = None,
    ) -> list[Path]:
        """Find all handler contract files under a directory.

        Searches for both handler_contract.yaml and contract.yaml files.
        Files exceeding MAX_CONTRACT_SIZE are skipped during discovery
        to fail fast before expensive path resolution and loading.

        Contract File Precedence:
            **WARNING**: When BOTH ``handler_contract.yaml`` AND ``contract.yaml``
            exist in the same directory, BOTH files are loaded as separate handlers.
            This is intentional to support different use cases:

            - ``handler_contract.yaml``: Dedicated handler contract (preferred)
            - ``contract.yaml``: General ONEX contract that may also define handlers

            If this behavior causes confusion or duplicate handler registrations,
            a warning is logged to alert operators. Best practice is to use only
            ONE contract file per handler directory to avoid ambiguity.

            See: docs/patterns/handler_plugin_loader.md#contract-file-precedence

        Args:
            directory: Directory to search recursively.
            correlation_id: Optional correlation ID for tracing and error context.
            max_handlers: Optional maximum number of handlers to discover.
                If specified, discovery stops after finding this many contract files.
                Propagated to file size validation for consistent traceability.

        Returns:
            List of paths to contract files that pass size validation.
        """
        contract_files: list[Path] = []
        # Track directories with both contract types to warn about ambiguity
        dirs_with_both_contracts: set[Path] = set()
        # Track if max_handlers limit was reached
        limit_reached = False

        # Search for valid contract filenames in a single scan
        # This consolidates two rglob() calls into one for better performance
        # WARNING: Both handler_contract.yaml and contract.yaml are loaded if present
        # in the same directory. This may lead to duplicate handler registrations
        # if both files define handlers. See docstring for details.
        valid_filenames = {HANDLER_CONTRACT_FILENAME, CONTRACT_YAML_FILENAME}
        for path in directory.rglob("*.yaml"):
            # Check if we've reached the max_handlers limit
            if max_handlers is not None and len(contract_files) >= max_handlers:
                limit_reached = True
                break

            if path.name in valid_filenames and path.is_file():
                # Early size validation to skip oversized files before expensive operations
                if (
                    self._validate_file_size(
                        path, correlation_id=correlation_id, raise_on_error=False
                    )
                    is None
                ):
                    continue

                contract_files.append(path)

        # Log warning if limit was reached
        if limit_reached:
            logger.warning(
                "Handler discovery limit reached: stopped at %d handlers. "
                "Increase max_handlers to discover more.",
                max_handlers,
                extra={
                    "max_handlers": max_handlers,
                    "directory": str(directory),
                    "correlation_id": correlation_id,
                },
            )

        # Detect directories with both contract types and warn about ambiguity
        # This is an O(n) check after discovery, not during, to avoid overhead
        # on every file. Build a map of parent_dir -> set of contract filenames.
        dir_to_contract_types: dict[Path, set[str]] = {}
        for path in contract_files:
            parent = path.parent
            if parent not in dir_to_contract_types:
                dir_to_contract_types[parent] = set()
            dir_to_contract_types[parent].add(path.name)

        # Warn for each directory that has both contract types
        for parent_dir, filenames in dir_to_contract_types.items():
            if len(filenames) > 1:
                logger.warning(
                    "AMBIGUOUS CONTRACT CONFIGURATION: Directory '%s' contains both "
                    "%s and %s. BOTH files will be loaded as separate handlers. "
                    "This may cause duplicate handler registrations or unexpected behavior. "
                    "Best practice: Use only ONE contract file per handler directory. "
                    "See: docs/patterns/handler_plugin_loader.md#contract-file-precedence",
                    parent_dir,
                    HANDLER_CONTRACT_FILENAME,
                    CONTRACT_YAML_FILENAME,
                    extra={
                        "directory": str(parent_dir),
                        "contract_files": sorted(filenames),
                        "correlation_id": correlation_id,
                        "severity": "configuration_warning",
                    },
                )

        # Deduplicate by resolved path
        seen: set[Path] = set()
        deduplicated: list[Path] = []
        for path in contract_files:
            # path.resolve() can raise OSError in several scenarios:
            # - Broken symlinks: symlink target no longer exists
            # - Race conditions: file deleted between glob discovery and resolution
            # - Permission issues: lacking read permission on parent directories
            # - Filesystem errors: unmounted volumes, network filesystem failures
            try:
                resolved = path.resolve()
            except OSError as e:
                logger.warning(
                    "Failed to resolve path %s: %s",
                    path,
                    e,
                    extra={"path": str(path), "error": str(e)},
                )
                continue
            if resolved not in seen:
                seen.add(resolved)
                deduplicated.append(path)

        return deduplicated


__all__ = [
    "CONTRACT_YAML_FILENAME",
    "HANDLER_CONTRACT_FILENAME",
    "HandlerPluginLoader",
    "MAX_CONTRACT_SIZE",
]
