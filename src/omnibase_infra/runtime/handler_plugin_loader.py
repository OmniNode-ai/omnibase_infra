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
    from multiple coroutines. Each load operation is independent.

See Also:
    - ProtocolHandlerPluginLoader: Protocol definition for plugin loaders
    - HandlerContractSource: Contract discovery and parsing
    - ModelLoadedHandler: Model representing loaded handler metadata

.. versionadded:: 0.7.0
    Created as part of OMN-1132 handler plugin loader implementation.
"""

from __future__ import annotations

import importlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_infra.enums import EnumHandlerTypeCategory, EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ProtocolConfigurationError
from omnibase_infra.models.errors import ModelInfraErrorContext
from omnibase_infra.models.runtime import ModelLoadedHandler
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

# Handler type category mapping from contract strings to enum
_HANDLER_TYPE_CATEGORY_MAP: dict[str, EnumHandlerTypeCategory] = {
    "compute": EnumHandlerTypeCategory.COMPUTE,
    "effect": EnumHandlerTypeCategory.EFFECT,
    "nondeterministic_compute": EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
}


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
        >>> handler = await loader.load_from_contract(
        ...     Path("src/handlers/auth/handler_contract.yaml")
        ... )
        >>> print(f"Loaded: {handler.handler_name}")

        >>> # Load all handlers from a directory
        >>> handlers = await loader.load_from_directory(Path("src/handlers"))
        >>> print(f"Loaded {len(handlers)} handlers")

        >>> # Discover with glob patterns
        >>> handlers = await loader.discover_and_load([
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

    async def load_from_contract(
        self,
        contract_path: Path,
    ) -> ModelLoadedHandler:
        """Load a single handler from a contract file.

        Parses the contract YAML file at the given path, validates it,
        imports the handler class, validates protocol compliance, and
        returns a ModelLoadedHandler with the loaded metadata.

        Args:
            contract_path: Path to the handler contract YAML file.
                Must be an absolute or relative path to an existing file.

        Returns:
            ModelLoadedHandler containing the loaded handler metadata
            including handler class, version, and contract information.

        Raises:
            ProtocolConfigurationError: If the contract file is invalid,
                missing required fields, or fails validation. Error codes:
                - HANDLER_LOADER_001: Contract file not found
                - HANDLER_LOADER_002: Invalid YAML syntax
                - HANDLER_LOADER_003: Schema validation failed
                - HANDLER_LOADER_004: Missing required fields
                - HANDLER_LOADER_005: Contract file exceeds size limit
                - HANDLER_LOADER_006: Handler does not implement protocol
            InfraConnectionError: If the handler class cannot be imported.
                Error codes:
                - HANDLER_LOADER_010: Module not found
                - HANDLER_LOADER_011: Class not found in module
                - HANDLER_LOADER_012: Import error (syntax/dependency)
        """
        logger.debug(
            "Loading handler from contract: %s",
            contract_path,
            extra={"contract_path": str(contract_path)},
        )

        # Validate contract path exists
        if not contract_path.exists():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                f"Contract file not found: {contract_path}",
                context=context,
                loader_error="HANDLER_LOADER_001",
                contract_path=str(contract_path),
            )

        if not contract_path.is_file():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                f"Contract path is not a file: {contract_path}",
                context=context,
                loader_error="HANDLER_LOADER_001",
                contract_path=str(contract_path),
            )

        # Validate file size
        try:
            file_size = contract_path.stat().st_size
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                f"Failed to stat contract file: {e}",
                context=context,
                loader_error="HANDLER_LOADER_001",
                contract_path=str(contract_path),
            ) from e

        if file_size > MAX_CONTRACT_SIZE:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                f"Contract file exceeds size limit: {file_size} bytes "
                f"(max: {MAX_CONTRACT_SIZE} bytes)",
                context=context,
                loader_error="HANDLER_LOADER_005",
                contract_path=str(contract_path),
                file_size=file_size,
                max_size=MAX_CONTRACT_SIZE,
            )

        # Parse YAML contract
        try:
            with contract_path.open("r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                f"Invalid YAML syntax in contract: {e}",
                context=context,
                loader_error="HANDLER_LOADER_002",
                contract_path=str(contract_path),
            ) from e
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                f"Failed to read contract file: {e}",
                context=context,
                loader_error="HANDLER_LOADER_001",
                contract_path=str(contract_path),
            ) from e

        if raw_data is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                "Contract file is empty",
                context=context,
                loader_error="HANDLER_LOADER_003",
                contract_path=str(contract_path),
            )

        # Extract required fields from contract
        handler_name = self._extract_handler_name(raw_data, contract_path)
        handler_class_path = self._extract_handler_class(raw_data, contract_path)
        handler_type = self._extract_handler_type(raw_data, contract_path)
        capability_tags = self._extract_capability_tags(raw_data)

        # Import and validate handler class
        handler_class = self._import_handler_class(handler_class_path, contract_path)

        # Validate handler implements protocol
        if not self._validate_handler_protocol(handler_class):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                f"Handler class {handler_class_path} does not implement "
                "ProtocolHandler (missing 'describe' method)",
                context=context,
                loader_error="HANDLER_LOADER_006",
                contract_path=str(contract_path),
                handler_class=handler_class_path,
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

    async def load_from_directory(
        self,
        directory: Path,
    ) -> list[ModelLoadedHandler]:
        """Load all handlers from contract files in a directory.

        Recursively scans the given directory for handler contract files
        (handler_contract.yaml or contract.yaml), loads each handler,
        and returns a list of successfully loaded handlers.

        Failed loads are logged but do not stop processing of other handlers.

        Args:
            directory: Path to the directory to scan for contract files.
                Must be an existing directory.

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
        logger.debug(
            "Loading handlers from directory: %s",
            directory,
            extra={"directory": str(directory)},
        )

        # Validate directory exists
        if not directory.exists():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_directory",
            )
            raise ProtocolConfigurationError(
                f"Directory not found: {directory}",
                context=context,
                loader_error="HANDLER_LOADER_020",
                directory=str(directory),
            )

        if not directory.is_dir():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_directory",
            )
            raise ProtocolConfigurationError(
                f"Path is not a directory: {directory}",
                context=context,
                loader_error="HANDLER_LOADER_022",
                directory=str(directory),
            )

        # Find all contract files
        contract_files = self._find_contract_files(directory)

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
        for contract_path in contract_files:
            try:
                handler = await self.load_from_contract(contract_path)
                handlers.append(handler)
            except (ProtocolConfigurationError, InfraConnectionError) as e:
                logger.warning(
                    "Failed to load handler from %s: %s",
                    contract_path,
                    str(e),
                    extra={
                        "contract_path": str(contract_path),
                        "error": str(e),
                    },
                )
                continue

        logger.info(
            "Loaded %d handlers from directory: %s",
            len(handlers),
            directory,
            extra={
                "directory": str(directory),
                "loaded_count": len(handlers),
                "total_contracts": len(contract_files),
            },
        )

        return handlers

    async def discover_and_load(
        self,
        patterns: list[str],
    ) -> list[ModelLoadedHandler]:
        """Discover contracts matching glob patterns and load handlers.

        Searches for contract files matching the given glob patterns,
        deduplicates matches, loads each handler, and returns a list
        of successfully loaded handlers.

        Args:
            patterns: List of glob patterns to match contract files.
                Supports standard glob syntax including ** for recursive.

        Returns:
            List of successfully loaded handlers. May be empty if no
            patterns match or all fail validation.

        Raises:
            ProtocolConfigurationError: If patterns list is empty.
                Error codes:
                - HANDLER_LOADER_030: Empty patterns list
        """
        logger.debug(
            "Discovering handlers with patterns: %s",
            patterns,
            extra={"patterns": patterns},
        )

        if not patterns:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="discover_and_load",
            )
            raise ProtocolConfigurationError(
                "Patterns list cannot be empty",
                context=context,
                loader_error="HANDLER_LOADER_030",
            )

        # Collect all matching contract files, deduplicated by resolved path
        discovered_paths: set[Path] = set()

        for pattern in patterns:
            # Use Path.cwd() as base for glob patterns
            matched_paths = list(Path.cwd().glob(pattern))
            for path in matched_paths:
                if path.is_file():
                    resolved = path.resolve()
                    discovered_paths.add(resolved)

        logger.debug(
            "Discovered %d unique contract files from %d patterns",
            len(discovered_paths),
            len(patterns),
            extra={
                "patterns": patterns,
                "discovered_count": len(discovered_paths),
            },
        )

        # Load each discovered contract (graceful mode)
        handlers: list[ModelLoadedHandler] = []
        for contract_path in sorted(discovered_paths):
            try:
                handler = await self.load_from_contract(contract_path)
                handlers.append(handler)
            except (ProtocolConfigurationError, InfraConnectionError) as e:
                logger.warning(
                    "Failed to load handler from %s: %s",
                    contract_path,
                    str(e),
                    extra={
                        "contract_path": str(contract_path),
                        "error": str(e),
                    },
                )
                continue

        logger.info(
            "Discovered and loaded %d handlers from %d patterns",
            len(handlers),
            len(patterns),
            extra={
                "patterns": patterns,
                "loaded_count": len(handlers),
                "discovered_count": len(discovered_paths),
            },
        )

        return handlers

    def _validate_handler_protocol(self, handler_class: type) -> bool:
        """Validate handler implements required protocol (ProtocolHandler).

        Uses duck typing to verify the handler class has the required
        methods for ProtocolHandler compliance.

        Args:
            handler_class: The handler class to validate.

        Returns:
            True if handler implements required protocol methods, False otherwise.
        """
        # Check for required ProtocolHandler method: describe()
        # Per ONEX conventions, protocol compliance is verified via duck typing
        return hasattr(handler_class, "describe") and callable(
            getattr(handler_class, "describe", None)
        )

    def _import_handler_class(
        self,
        class_path: str,
        contract_path: Path,
    ) -> type[ProtocolHandler]:
        """Dynamically import handler class from fully qualified path.

        Args:
            class_path: Fully qualified class path (e.g., 'myapp.handlers.AuthHandler').
            contract_path: Path to the contract file (for error context).

        Returns:
            The imported handler class.

        Raises:
            InfraConnectionError: If the module or class cannot be imported.
        """
        # Split class path into module and class name
        if "." not in class_path:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
            )
            raise InfraConnectionError(
                f"Invalid class path '{class_path}': must be fully qualified "
                "(e.g., 'myapp.handlers.AuthHandler')",
                context=context,
                loader_error="HANDLER_LOADER_010",
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
            )
            raise InfraConnectionError(
                f"Module not found: {module_path}",
                context=context,
                loader_error="HANDLER_LOADER_010",
                module_path=module_path,
                class_path=class_path,
                contract_path=str(contract_path),
            ) from e
        except ImportError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
            )
            raise InfraConnectionError(
                f"Import error loading module {module_path}: {e}",
                context=context,
                loader_error="HANDLER_LOADER_012",
                module_path=module_path,
                class_path=class_path,
                contract_path=str(contract_path),
            ) from e

        # Get the class from the module
        if not hasattr(module, class_name):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
            )
            raise InfraConnectionError(
                f"Class '{class_name}' not found in module '{module_path}'",
                context=context,
                loader_error="HANDLER_LOADER_011",
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
            )
            raise InfraConnectionError(
                f"'{class_path}' is not a class",
                context=context,
                loader_error="HANDLER_LOADER_011",
                class_path=class_path,
                contract_path=str(contract_path),
            )

        return handler_class  # type: ignore[return-value]

    def _find_contract_files(self, directory: Path) -> list[Path]:
        """Find all handler contract files under a directory.

        Searches for both handler_contract.yaml and contract.yaml files.

        Args:
            directory: Directory to search recursively.

        Returns:
            List of paths to contract files.
        """
        contract_files: list[Path] = []

        # Search for handler_contract.yaml files (exact case-sensitive match)
        for path in directory.rglob(HANDLER_CONTRACT_FILENAME):
            if path.name == HANDLER_CONTRACT_FILENAME and path.is_file():
                contract_files.append(path)

        # Search for contract.yaml files (exact case-sensitive match)
        for path in directory.rglob(CONTRACT_YAML_FILENAME):
            if path.name == CONTRACT_YAML_FILENAME and path.is_file():
                contract_files.append(path)

        # Deduplicate by resolved path
        seen: set[Path] = set()
        deduplicated: list[Path] = []
        for path in contract_files:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                deduplicated.append(path)

        return deduplicated

    def _extract_handler_name(
        self,
        raw_data: dict[str, object],
        contract_path: Path,
    ) -> str:
        """Extract handler name from contract data.

        Args:
            raw_data: Parsed YAML contract data.
            contract_path: Path to contract file (for error context).

        Returns:
            Handler name string.

        Raises:
            ProtocolConfigurationError: If handler_name/name is missing or invalid.
        """
        # Try handler_name first, then name
        handler_name = raw_data.get("handler_name") or raw_data.get("name")

        if handler_name is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="extract_handler_name",
            )
            raise ProtocolConfigurationError(
                "Contract missing required field: 'handler_name' or 'name'",
                context=context,
                loader_error="HANDLER_LOADER_004",
                contract_path=str(contract_path),
            )

        if not isinstance(handler_name, str) or not handler_name.strip():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="extract_handler_name",
            )
            raise ProtocolConfigurationError(
                "Contract field 'handler_name' must be a non-empty string",
                context=context,
                loader_error="HANDLER_LOADER_004",
                contract_path=str(contract_path),
            )

        return handler_name.strip()

    def _extract_handler_class(
        self,
        raw_data: dict[str, object],
        contract_path: Path,
    ) -> str:
        """Extract handler class path from contract data.

        Args:
            raw_data: Parsed YAML contract data.
            contract_path: Path to contract file (for error context).

        Returns:
            Fully qualified handler class path.

        Raises:
            ProtocolConfigurationError: If handler_class is missing or invalid.
        """
        handler_class = raw_data.get("handler_class")

        if handler_class is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="extract_handler_class",
            )
            raise ProtocolConfigurationError(
                "Contract missing required field: 'handler_class'",
                context=context,
                loader_error="HANDLER_LOADER_004",
                contract_path=str(contract_path),
            )

        if not isinstance(handler_class, str) or not handler_class.strip():
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="extract_handler_class",
            )
            raise ProtocolConfigurationError(
                "Contract field 'handler_class' must be a non-empty string",
                context=context,
                loader_error="HANDLER_LOADER_004",
                contract_path=str(contract_path),
            )

        return handler_class.strip()

    def _extract_handler_type(
        self,
        raw_data: dict[str, object],
        contract_path: Path,
    ) -> EnumHandlerTypeCategory:
        """Extract handler type category from contract data.

        Args:
            raw_data: Parsed YAML contract data.
            contract_path: Path to contract file (for error context).

        Returns:
            Handler type category enum value.

        Raises:
            ProtocolConfigurationError: If handler_type is missing or invalid.
        """
        handler_type = raw_data.get("handler_type")

        if handler_type is None:
            # Default to EFFECT if not specified (conservative default)
            return EnumHandlerTypeCategory.EFFECT

        if not isinstance(handler_type, str):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="extract_handler_type",
            )
            raise ProtocolConfigurationError(
                "Contract field 'handler_type' must be a string",
                context=context,
                loader_error="HANDLER_LOADER_003",
                contract_path=str(contract_path),
            )

        handler_type_lower = handler_type.lower().strip()

        if handler_type_lower not in _HANDLER_TYPE_CATEGORY_MAP:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="extract_handler_type",
            )
            valid_types = ", ".join(_HANDLER_TYPE_CATEGORY_MAP.keys())
            raise ProtocolConfigurationError(
                f"Invalid handler_type '{handler_type}'. Valid types: {valid_types}",
                context=context,
                loader_error="HANDLER_LOADER_003",
                contract_path=str(contract_path),
                handler_type=handler_type,
                valid_types=list(_HANDLER_TYPE_CATEGORY_MAP.keys()),
            )

        return _HANDLER_TYPE_CATEGORY_MAP[handler_type_lower]

    def _extract_capability_tags(
        self,
        raw_data: dict[str, object],
    ) -> list[str]:
        """Extract capability tags from contract data.

        Args:
            raw_data: Parsed YAML contract data.

        Returns:
            List of capability tag strings. Empty list if not specified.
        """
        tags = raw_data.get("capability_tags") or raw_data.get("tags")

        if tags is None:
            return []

        if isinstance(tags, list):
            # Filter to only string tags, skip invalid types silently
            return [str(tag) for tag in tags if isinstance(tag, str)]

        # Single tag as string
        if isinstance(tags, str):
            return [tags]

        return []


__all__ = [
    "CONTRACT_YAML_FILENAME",
    "HANDLER_CONTRACT_FILENAME",
    "HandlerPluginLoader",
    "MAX_CONTRACT_SIZE",
]
