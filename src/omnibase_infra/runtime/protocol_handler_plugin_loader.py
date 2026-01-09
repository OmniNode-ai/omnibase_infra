# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol interface for Handler Plugin Loaders in ONEX Infrastructure.

This module defines the ProtocolHandlerPluginLoader interface for discovering
and loading handlers from contract files. Handler plugin loaders scan filesystem
paths for handler contracts, validate them, and register them with the handler
registry.

Part of OMN-1132: Handler Plugin Loader implementation.

Thread Safety:
    Implementations may be invoked concurrently from multiple threads.
    Implementations SHOULD be stateless or use appropriate synchronization
    primitives for any shared mutable state.

Example Usage:
    ```python
    from pathlib import Path
    from omnibase_infra.runtime.protocol_handler_plugin_loader import (
        ProtocolHandlerPluginLoader,
    )

    class FileSystemHandlerPluginLoader:
        '''Loads handlers from filesystem contract files.'''

        def load_from_contract(
            self,
            contract_path: Path,
            correlation_id: str | None = None,
        ) -> ModelLoadedHandler:
            # Parse contract YAML and load handler class
            ...

        def load_from_directory(
            self,
            directory: Path,
            correlation_id: str | None = None,
        ) -> list[ModelLoadedHandler]:
            # Scan directory for contracts and load all handlers
            ...

        def discover_and_load(
            self,
            patterns: list[str],
            correlation_id: str | None = None,
        ) -> list[ModelLoadedHandler]:
            # Match glob patterns and load discovered handlers
            ...

    # Type checking works via Protocol
    loader: ProtocolHandlerPluginLoader = FileSystemHandlerPluginLoader()
    handlers = loader.load_from_directory(Path("src/handlers"))
    ```

See Also:
    - ProtocolContractSource: Protocol for handler contract sources
    - ProtocolContractDescriptor: Protocol for handler descriptors
    - ModelLoadedHandler: Model representing a successfully loaded handler

.. versionadded:: 0.6.4
    Created as part of OMN-1132 handler plugin loader implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.models.runtime import ModelLoadedHandler


@runtime_checkable
class ProtocolHandlerPluginLoader(Protocol):
    """Protocol for loading handlers as plugins from contract files.

    Handler plugin loaders discover handler contracts from filesystem paths,
    validate the contracts against the handler schema, import the handler
    classes, and register them with the handler registry.

    This protocol enables:
        - Dependency injection of handler loading strategies
        - Test isolation through mock loaders
        - Runtime swapping of discovery mechanisms
        - Composition of multiple loader implementations

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing
        rather than isinstance checks. Verify required methods exist:

        .. code-block:: python

            # Verify required methods exist
            if (hasattr(loader, 'load_from_contract') and
                hasattr(loader, 'load_from_directory') and
                hasattr(loader, 'discover_and_load') and
                callable(loader.load_from_contract)):
                registry.set_loader(loader)
            else:
                raise TypeError("Object does not implement ProtocolHandlerPluginLoader")

    Attributes:
        None - this is a pure protocol with no required properties.

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    .. versionadded:: 0.6.4
    """

    def load_from_contract(
        self,
        contract_path: Path,
        correlation_id: str | None = None,
    ) -> ModelLoadedHandler:
        """Load a single handler from a contract file.

        Parses the contract YAML file at the given path, validates it against
        the handler contract schema, imports the handler class, and returns
        a ModelLoadedHandler containing the loaded handler metadata.

        Args:
            contract_path: Path to the handler contract YAML file. Must be an
                absolute or relative path to an existing file with .yaml or
                .yml extension.
            correlation_id: Optional correlation ID for tracing and error context.

        Returns:
            ModelLoadedHandler: Container with the loaded handler metadata
                including handler class, version, and contract information.

        Raises:
            ProtocolConfigurationError: If the contract file is invalid, missing
                required fields, or fails schema validation. Error codes:
                - HANDLER_LOADER_001: Contract file not found
                - HANDLER_LOADER_002: Invalid YAML syntax
                - HANDLER_LOADER_003: Schema validation failed
                - HANDLER_LOADER_004: Missing required fields
            InfraConnectionError: If the handler class cannot be imported due
                to module resolution failures or import errors. Error codes:
                - HANDLER_LOADER_010: Module not found
                - HANDLER_LOADER_011: Class not found in module
                - HANDLER_LOADER_012: Import error (syntax/dependency)

        Example:
            .. code-block:: python

                loader: ProtocolHandlerPluginLoader = FileSystemHandlerPluginLoader()
                handler = loader.load_from_contract(
                    Path("src/handlers/user_handler/contract.yaml")
                )
                print(f"Loaded handler: {handler.handler_id}")
        """
        ...

    def load_from_directory(
        self,
        directory: Path,
        correlation_id: str | None = None,
    ) -> list[ModelLoadedHandler]:
        """Load all handlers from contract files in a directory.

        Recursively scans the given directory for handler contract files
        (contract.yaml or contract.yml), loads each handler, and returns
        a list of successfully loaded handlers.

        Failed loads are logged but do not stop processing of other handlers.
        Use strict mode configuration to change this behavior.

        Args:
            directory: Path to the directory to scan for contract files.
                Must be an existing directory. Subdirectories are scanned
                recursively.
            correlation_id: Optional correlation ID for tracing and error context.

        Returns:
            list[ModelLoadedHandler]: List of successfully loaded handlers.
                May be empty if no contracts are found or all fail validation.
                Order is not guaranteed.

        Raises:
            ProtocolConfigurationError: If the directory does not exist or
                is not accessible. Error codes:
                - HANDLER_LOADER_020: Directory not found
                - HANDLER_LOADER_021: Permission denied
                - HANDLER_LOADER_022: Not a directory

        Example:
            .. code-block:: python

                loader: ProtocolHandlerPluginLoader = FileSystemHandlerPluginLoader()
                handlers = loader.load_from_directory(
                    Path("src/omnibase_infra/handlers")
                )
                print(f"Loaded {len(handlers)} handlers")
                for handler in handlers:
                    print(f"  - {handler.handler_id}")
        """
        ...

    def discover_and_load(
        self,
        patterns: list[str],
        correlation_id: str | None = None,
    ) -> list[ModelLoadedHandler]:
        """Discover contracts matching glob patterns and load handlers.

        Searches for contract files matching the given glob patterns,
        deduplicates matches, loads each handler, and returns a list
        of successfully loaded handlers.

        This method supports both absolute and relative glob patterns.
        Relative patterns are resolved from the current working directory.

        Args:
            patterns: List of glob patterns to match contract files.
                Supports standard glob syntax including:
                - ``*`` - matches any characters except path separator
                - ``**`` - matches any characters including path separator
                - ``?`` - matches single character
                - ``[seq]`` - matches any character in seq

                Common patterns:
                - ``src/**/contract.yaml`` - all contracts under src
                - ``handlers/*/contract.yaml`` - direct subdirs only
                - ``**/*.handler.yaml`` - alternative naming convention
            correlation_id: Optional correlation ID for tracing and error context.

        Returns:
            list[ModelLoadedHandler]: List of successfully loaded handlers.
                May be empty if no patterns match or all fail validation.
                Duplicate matches (same file from multiple patterns) are
                loaded only once.

        Raises:
            ProtocolConfigurationError: If patterns list is empty or contains
                invalid pattern syntax. Error codes:
                - HANDLER_LOADER_030: Empty patterns list
                - HANDLER_LOADER_031: Invalid glob pattern syntax

        Example:
            .. code-block:: python

                loader: ProtocolHandlerPluginLoader = FileSystemHandlerPluginLoader()
                handlers = loader.discover_and_load([
                    "src/omnibase_infra/handlers/**/contract.yaml",
                    "src/omnibase_infra/nodes/**/contract.yaml",
                ])
                print(f"Discovered and loaded {len(handlers)} handlers")
        """
        ...


__all__: list[str] = [
    "ProtocolHandlerPluginLoader",
]
