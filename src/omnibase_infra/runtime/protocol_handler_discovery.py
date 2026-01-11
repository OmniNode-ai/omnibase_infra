# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol interface for Handler Discovery Services in ONEX Infrastructure.

This module defines the ProtocolHandlerDiscovery interface for discovering
handlers from contract files and registering them with the runtime. Handler
discovery services scan filesystem paths for handler contracts, validate them,
load the handler classes, and register them for use.

Part of OMN-1133: Handler Discovery Service implementation.

Thread Safety:
    Implementations may be invoked concurrently from multiple threads.
    Implementations SHOULD be stateless or use appropriate synchronization
    primitives for any shared mutable state.

Example Usage:
    ```python
    from pathlib import Path
    from uuid import UUID
    from omnibase_infra.runtime.protocol_handler_discovery import (
        ProtocolHandlerDiscovery,
    )

    class FileSystemHandlerDiscovery:
        '''Discovers handlers from filesystem contract files.'''

        async def discover_and_register(
            self,
            contract_paths: list[Path],
            correlation_id: UUID | None = None,
        ) -> ModelDiscoveryResult:
            # Scan paths for contracts, load handlers, register them
            ...

    # Type checking works via Protocol
    discovery: ProtocolHandlerDiscovery = FileSystemHandlerDiscovery()
    result = await discovery.discover_and_register([Path("src/handlers")])
    print(f"Discovered: {result.handlers_discovered}, Registered: {result.handlers_registered}")
    ```

See Also:
    - ProtocolHandlerPluginLoader: Protocol for loading individual handlers
    - ProtocolContractSource: Protocol for handler contract sources
    - ModelDiscoveryResult: Model representing discovery operation results

.. versionadded:: 0.7.0
    Created as part of OMN-1133 handler discovery service implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.models.runtime.model_discovery_result import (
        ModelDiscoveryResult,
    )


@runtime_checkable
class ProtocolHandlerDiscovery(Protocol):
    """Protocol for handler discovery services.

    Implementations discover handlers from contract files and register them
    with the runtime. This enables contract-driven handler loading without
    manual wiring.

    Pattern: Duck typing - implementations don't need to inherit from this protocol,
    they just need to implement the required methods.

    This protocol enables:
        - Dependency injection of handler discovery strategies
        - Test isolation through mock discovery services
        - Runtime swapping of discovery mechanisms
        - Composition of multiple discovery implementations

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing
        rather than isinstance checks. Verify required methods exist:

        .. code-block:: python

            # Verify required methods exist
            if (hasattr(discovery, 'discover_and_register') and
                callable(discovery.discover_and_register)):
                runtime.set_discovery_service(discovery)
            else:
                raise TypeError("Object does not implement ProtocolHandlerDiscovery")

    Attributes:
        None - this is a pure protocol with no required properties.

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    .. versionadded:: 0.7.0
    """

    async def discover_and_register(
        self,
        contract_paths: list[Path],
        correlation_id: UUID | None = None,
    ) -> ModelDiscoveryResult:
        """Discover handlers from contracts and register them.

        Scans the provided paths for handler contract files, validates the
        contracts against the handler schema, imports the handler classes,
        and registers them with the runtime handler registry.

        This is the primary entry point for contract-driven handler discovery.
        It combines the functionality of:
        - Path scanning (directories are scanned recursively)
        - Contract file discovery (finds handler_contract.yaml or contract.yaml)
        - Handler loading (imports handler classes from contracts)
        - Handler registration (registers loaded handlers with runtime)

        Args:
            contract_paths: List of paths to scan for handler contracts.
                Can be directories (recursive scan) or direct contract files.

                **Directory paths**: Scanned recursively for contract files.
                The scanner looks for ``handler_contract.yaml`` (preferred) or
                ``contract.yaml`` files.

                **File paths**: Treated as direct contract files. Must be valid
                YAML files with ``.yaml`` or ``.yml`` extension.

                **Mixed paths**: Both directories and files can be provided
                in the same call.

            correlation_id: Optional correlation ID for tracing and error
                context. If not provided, a UUID4 is auto-generated. This ID
                is propagated to all sub-operations for distributed tracing.

        Returns:
            ModelDiscoveryResult: Container with discovery operation results
                including:

                - ``handlers_discovered``: Number of handlers found during discovery
                - ``handlers_registered``: Number of handlers successfully registered
                - ``errors``: List of ModelDiscoveryError for failed handlers
                - ``warnings``: List of ModelDiscoveryWarning for non-fatal issues
                - ``discovered_at``: Timestamp when discovery completed

        Raises:
            ProtocolConfigurationError: If critical configuration issues prevent
                discovery from proceeding. Error codes:

                - DISCOVERY_001: Empty contract_paths list provided
                - DISCOVERY_002: All provided paths are invalid (none exist)
                - DISCOVERY_003: Configuration prevents any discovery

                Note that individual path or contract failures do NOT raise
                exceptions - they are captured in the result's ``errors`` list
                to allow partial success.

        Example:
            .. code-block:: python

                from pathlib import Path
                from uuid import uuid4

                discovery: ProtocolHandlerDiscovery = FileSystemHandlerDiscovery()

                # Discover from multiple directories
                result = await discovery.discover_and_register(
                    contract_paths=[
                        Path("src/omnibase_infra/handlers"),
                        Path("src/omnibase_infra/nodes"),
                    ],
                    correlation_id=uuid4(),
                )

                print(f"Discovered: {result.handlers_discovered}")
                print(f"Registered: {result.handlers_registered}")

                # Check for errors
                if result.errors:
                    for error in result.errors:
                        print(f"Error: {error.contract_path} - {error.message}")

                # Check for warnings
                if result.warnings:
                    for warning in result.warnings:
                        print(f"Warning: {warning}")

            .. code-block:: python

                # Mixed paths: directories and direct contract files
                result = await discovery.discover_and_register(
                    contract_paths=[
                        Path("src/handlers"),  # Directory - scanned recursively
                        Path("plugins/custom_handler/contract.yaml"),  # Direct file
                    ],
                )

        Thread Safety:
            This method may be called concurrently from multiple threads or
            coroutines. Implementations MUST ensure thread-safe access to
            shared state (e.g., handler registries).

        Performance:
            Discovery operations may involve significant I/O (filesystem
            scanning, YAML parsing) and CPU (handler class loading). For
            large codebases, consider:

            - Limiting scan depth via configuration
            - Caching discovery results between restarts
            - Running discovery during startup, not per-request
        """
        ...


__all__: list[str] = [
    "ProtocolHandlerDiscovery",
]
