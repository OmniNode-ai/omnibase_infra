# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Publisher Source Protocol.

Defines the protocol interface for contract sources. Sources are responsible
for discovering contracts from their respective backends (filesystem, package
resources, etc.).

This protocol is distinct from ProtocolContractSource (used for handler
discovery) because it returns raw contracts for publishing rather than
parsed handler descriptors.

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.services.contract_publisher.sources.model_discovered import (
    ModelDiscoveredContract,
)


@runtime_checkable
class ProtocolContractPublisherSource(Protocol):
    """Protocol for contract publisher sources.

    Defines the interface for sources that discover contracts for bulk
    publishing. Implementations must provide discovery of contracts
    from their specific backend.

    Methods:
        source_type: Returns identifier for the source type
        source_description: Returns human-readable description
        discover_contracts: Discovers all contracts from the source

    Implementations:
        - SourceContractFilesystem: Discovers from directory tree
        - SourceContractPackage: Discovers from package resources
        - SourceContractComposite: Merges multiple sources

    Example:
        >>> class MySource:
        ...     @property
        ...     def source_type(self) -> str:
        ...         return "custom"
        ...
        ...     @property
        ...     def source_description(self) -> str:
        ...         return "custom: my-source"
        ...
        ...     async def discover_contracts(self) -> list[ModelDiscoveredContract]:
        ...         return []

    .. versionadded:: 0.3.0
    """

    @property
    def source_type(self) -> str:
        """Return source type identifier.

        Used for logging and statistics. Should be a short, lowercase
        identifier like "filesystem" or "package".

        Returns:
            Source type identifier string
        """
        ...

    @property
    def source_description(self) -> str:
        """Return human-readable source description.

        Used for error messages and logging. Should include the source
        type and any relevant configuration details.

        Returns:
            Human-readable description (e.g., "filesystem: /app/contracts")
        """
        ...

    async def discover_contracts(self) -> list[ModelDiscoveredContract]:
        """Discover all contracts from this source.

        Searches the source backend for contract.yaml files and returns
        them as ModelDiscoveredContract instances. The handler_id and
        content_hash fields are NOT populated at this stage - they are
        filled in during validation.

        Returns:
            List of discovered contracts with origin, ref, and text populated

        Raises:
            OSError: If the source location is inaccessible
        """
        ...


__all__ = ["ProtocolContractPublisherSource"]
