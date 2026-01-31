# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Contract Source.

Discovers contracts from a directory tree by globbing for contract.yaml files.
This is the primary source for local development and CI/CD environments.

Discovery Pattern:
    Globs for **/contract.yaml under the configured root directory.

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from __future__ import annotations

import logging
from pathlib import Path

from omnibase_infra.services.contract_publisher.sources.model_discovered import (
    ModelDiscoveredContract,
)

logger = logging.getLogger(__name__)


class SourceContractFilesystem:
    """Filesystem-based contract source.

    Recursively scans a directory for contract.yaml files and returns
    them as ModelDiscoveredContract instances.

    Discovery:
        Uses Path.glob("**/contract.yaml") to find all contract files
        under the root directory.

    Error Handling:
        - If root doesn't exist: Returns empty list (logged as debug)
        - If file read fails: Logs warning, skips file, continues

    Attributes:
        _root: Root directory for discovery

    Example:
        >>> source = SourceContractFilesystem(Path("/app/contracts"))
        >>> contracts = await source.discover_contracts()
        >>> for contract in contracts:
        ...     print(f"Found: {contract.ref}")

    .. versionadded:: 0.3.0
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        """Initialize filesystem source.

        Args:
            root: Root directory for contract discovery
        """
        self._root = root

    @property
    def source_type(self) -> str:
        """Return source type identifier.

        Returns:
            "filesystem"
        """
        return "filesystem"

    @property
    def source_description(self) -> str:
        """Return human-readable source description.

        Returns:
            Description including the root path
        """
        return f"filesystem: {self._root}"

    @property
    def root(self) -> Path:
        """Return the root directory.

        Returns:
            Root directory Path
        """
        return self._root

    async def discover_contracts(self) -> list[ModelDiscoveredContract]:
        """Discover all contracts from the filesystem.

        Globs for **/contract.yaml under the root directory and reads
        each file's contents.

        Returns:
            List of discovered contracts with origin="filesystem"

        Note:
            Returns empty list if root doesn't exist (not an error).
            Individual file read failures are logged and skipped.

        Note:
            This method is ``async`` for protocol consistency with
            :class:`ProtocolContractPublisherSource`. The underlying file I/O
            is synchronous, which is acceptable for this use case as contract
            discovery is an infrequent startup operation.
        """
        if not self._root.exists():
            logger.debug(
                "Filesystem source root does not exist: %s",
                self._root,
            )
            return []

        if not self._root.is_dir():
            logger.warning(
                "Filesystem source root is not a directory: %s",
                self._root,
            )
            return []

        contracts: list[ModelDiscoveredContract] = []

        # Glob for contract.yaml files
        for contract_path in self._root.glob("**/contract.yaml"):
            try:
                text = contract_path.read_text(encoding="utf-8")
                contract = ModelDiscoveredContract(
                    origin="filesystem",
                    ref=contract_path,
                    text=text,
                )
                contracts.append(contract)

                logger.debug(
                    "Discovered contract: %s",
                    contract_path,
                )

            except OSError as e:
                logger.warning(
                    "Failed to read contract file %s: %s",
                    contract_path,
                    e,
                )
                # Continue with other files

        logger.info(
            "Filesystem discovery complete: found %d contracts in %s",
            len(contracts),
            self._root,
        )

        return contracts


__all__ = ["SourceContractFilesystem"]
