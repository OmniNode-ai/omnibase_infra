# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Package Resource Contract Source.

Discovers contracts from installed Python package resources using
importlib.resources. This is the primary source for containerized
deployments where contracts ship with the wheel.

Discovery Pattern:
    Traverses package resources looking for contract.yaml files.

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from __future__ import annotations

import logging
from importlib import resources
from importlib.resources.abc import Traversable

from omnibase_infra.services.contract_publisher.sources.model_discovered import (
    ModelDiscoveredContract,
)

logger = logging.getLogger(__name__)


class SourceContractPackage:
    """Package resource-based contract source.

    Discovers contracts from installed Python package resources using
    importlib.resources API. Recursively traverses the package looking
    for contract.yaml files.

    Requirements:
        - The package must be installed (importable)
        - Contract files must be named "contract.yaml"
        - Package must include resources (MANIFEST.in or package_data)

    Attributes:
        _package_module: Fully qualified module name (e.g., "myapp.contracts")

    Example:
        >>> source = SourceContractPackage("myapp.contracts.handlers")
        >>> contracts = await source.discover_contracts()
        >>> for contract in contracts:
        ...     print(f"Found: {contract.ref}")

    .. versionadded:: 0.3.0
    """

    __slots__ = ("_package_module",)

    def __init__(self, package_module: str) -> None:
        """Initialize package source.

        Args:
            package_module: Fully qualified module name for contract discovery
                           (e.g., "myapp.contracts" or "myapp.contracts.handlers")
        """
        self._package_module = package_module

    @property
    def source_type(self) -> str:
        """Return source type identifier.

        Returns:
            "package"
        """
        return "package"

    @property
    def source_description(self) -> str:
        """Return human-readable source description.

        Returns:
            Description including the package module
        """
        return f"package: {self._package_module}"

    @property
    def package_module(self) -> str:
        """Return the package module name.

        Returns:
            Package module name
        """
        return self._package_module

    async def discover_contracts(self) -> list[ModelDiscoveredContract]:
        """Discover all contracts from the package.

        Uses importlib.resources to traverse the package and find
        contract.yaml files.

        Returns:
            List of discovered contracts with origin="package"

        Note:
            Returns empty list if package is not installed or has no resources.
            Individual resource read failures are logged and skipped.

        Note:
            This method is ``async`` for protocol consistency with
            :class:`ProtocolContractPublisherSource`. The underlying package
            resource traversal is synchronous, which is acceptable for this
            use case as contract discovery is an infrequent startup operation.
        """
        contracts: list[ModelDiscoveredContract] = []

        try:
            # Get the package root as a Traversable
            package_root = resources.files(self._package_module)

            # Recursively discover contracts
            self._discover_recursive(package_root, "", contracts)

        except ModuleNotFoundError:
            logger.debug(
                "Package not found: %s",
                self._package_module,
            )
            return []

        except TypeError as e:
            # resources.files() can raise TypeError for invalid packages
            logger.warning(
                "Invalid package for resource discovery: %s - %s",
                self._package_module,
                e,
            )
            return []

        logger.info(
            "Package discovery complete: found %d contracts in %s",
            len(contracts),
            self._package_module,
        )

        return contracts

    def _discover_recursive(
        self,
        traversable: Traversable,
        path_prefix: str,
        contracts: list[ModelDiscoveredContract],
    ) -> None:
        """Recursively discover contracts from a Traversable.

        Args:
            traversable: Current package resource to traverse
            path_prefix: Path prefix for building resource paths
            contracts: List to append discovered contracts to
        """
        try:
            # Check if this is a directory
            if not traversable.is_dir():
                return

            # Iterate children
            for child in traversable.iterdir():
                child_name = child.name
                child_path = (
                    f"{path_prefix}/{child_name}" if path_prefix else child_name
                )

                if child.is_file() and child_name == "contract.yaml":
                    # Found a contract file
                    try:
                        text = child.read_text(encoding="utf-8")
                        resource_path = f"{self._package_module}:{child_path}"

                        contract = ModelDiscoveredContract(
                            origin="package",
                            ref=resource_path,
                            text=text,
                        )
                        contracts.append(contract)

                        logger.debug(
                            "Discovered contract: %s",
                            resource_path,
                        )

                    except OSError as e:
                        logger.warning(
                            "Failed to read package resource %s: %s",
                            child_path,
                            e,
                        )

                elif child.is_dir():
                    # Recurse into subdirectory
                    self._discover_recursive(child, child_path, contracts)

        except OSError as e:
            logger.warning(
                "Error traversing package %s at %s: %s",
                self._package_module,
                path_prefix,
                e,
            )


__all__ = ["SourceContractPackage"]
