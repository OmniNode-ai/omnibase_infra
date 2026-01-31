# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Composite Contract Source.

Merges contracts from filesystem and package sources with deterministic
conflict detection. Both sources are discovered, and contracts are merged
by handler_id.

Merge Rules:
    1. Discover from both configured sources
    2. For each contract, parse handler_id and compute content_hash
    3. If same handler_id appears in both sources:
       - Same hash → dedup silently (use first occurrence)
       - Different hash → ModelContractError("duplicate_conflict")

Ordering:
    Results are sorted by (handler_id, origin, ref) for deterministic
    ordering across runs regardless of filesystem order.

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.services.contract_publisher.models import ModelContractError
from omnibase_infra.services.contract_publisher.sources.model_discovered import (
    ModelDiscoveredContract,
)

if TYPE_CHECKING:
    from omnibase_infra.services.contract_publisher.sources.source_filesystem import (
        SourceContractFilesystem,
    )
    from omnibase_infra.services.contract_publisher.sources.source_package import (
        SourceContractPackage,
    )

logger = logging.getLogger(__name__)


class SourceContractComposite:
    """Composite source that merges filesystem and package sources.

    Discovers contracts from both configured sources and merges them
    with deterministic conflict detection. At least one source must
    be configured.

    Conflict Detection:
        When the same handler_id appears in both sources:
        - If content hash matches: Deduplicate silently (keep first)
        - If content hash differs: Generate ModelContractError

    Ordering:
        All results are sorted by (handler_id, origin, ref) to ensure
        deterministic ordering across runs.

    Attributes:
        _filesystem_source: Optional filesystem source
        _package_source: Optional package source
        _last_merge_errors: Errors from last discover_contracts call

    Example:
        >>> filesystem = SourceContractFilesystem(Path("/app/contracts"))
        >>> package = SourceContractPackage("myapp.contracts")
        >>> composite = SourceContractComposite(filesystem, package)
        >>> contracts = await composite.discover_contracts()
        >>> errors = composite.get_merge_errors()

    .. versionadded:: 0.3.0
    """

    __slots__ = ("_filesystem_source", "_last_merge_errors", "_package_source")

    def __init__(
        self,
        filesystem_source: SourceContractFilesystem | None = None,
        package_source: SourceContractPackage | None = None,
    ) -> None:
        """Initialize composite source.

        Args:
            filesystem_source: Optional filesystem source
            package_source: Optional package source

        Raises:
            ValueError: If neither source is configured
        """
        if not filesystem_source and not package_source:
            raise ValueError(
                "Composite source requires at least one source "
                "(filesystem_source or package_source)"
            )

        self._filesystem_source = filesystem_source
        self._package_source = package_source
        self._last_merge_errors: list[ModelContractError] = []

    @property
    def source_type(self) -> str:
        """Return source type identifier.

        Returns:
            "composite"
        """
        return "composite"

    @property
    def source_description(self) -> str:
        """Return human-readable source description.

        Returns:
            Description including configured sources
        """
        parts: list[str] = ["composite:"]
        if self._filesystem_source:
            parts.append(f"filesystem={self._filesystem_source.root}")
        if self._package_source:
            parts.append(f"package={self._package_source.package_module}")
        return " ".join(parts)

    @property
    def filesystem_source(self) -> SourceContractFilesystem | None:
        """Return the filesystem source if configured."""
        return self._filesystem_source

    @property
    def package_source(self) -> SourceContractPackage | None:
        """Return the package source if configured."""
        return self._package_source

    async def discover_contracts(self) -> list[ModelDiscoveredContract]:
        """Discover contracts from all configured sources and merge.

        Discovers from both sources (if configured), then merges with
        conflict detection. Conflict errors are stored internally and
        can be retrieved via get_merge_errors().

        Returns:
            Deduplicated and sorted contracts (excludes conflicts)

        Note:
            The returned contracts have content_hash populated.
            Call get_merge_errors() after discovery to get conflict errors.
        """
        all_contracts: list[ModelDiscoveredContract] = []

        # Clear previous errors
        self._last_merge_errors = []

        # Discover from filesystem source
        if self._filesystem_source:
            filesystem_contracts = await self._filesystem_source.discover_contracts()
            all_contracts.extend(filesystem_contracts)
            logger.debug(
                "Composite: discovered %d contracts from filesystem",
                len(filesystem_contracts),
            )

        # Discover from package source
        if self._package_source:
            package_contracts = await self._package_source.discover_contracts()
            all_contracts.extend(package_contracts)
            logger.debug(
                "Composite: discovered %d contracts from package",
                len(package_contracts),
            )

        # Compute content hashes for all contracts
        all_contracts = [c.with_content_hash() for c in all_contracts]

        # Merge with conflict detection
        merged, conflicts = self._merge_contracts(all_contracts)

        # Store errors for later retrieval
        self._last_merge_errors = conflicts

        # Sort merged contracts for deterministic ordering
        merged.sort(key=lambda c: c.sort_key())

        logger.info(
            "Composite discovery complete: %d merged, %d conflicts, %d total",
            len(merged),
            len(conflicts),
            len(all_contracts),
        )

        return merged

    def get_merge_errors(self) -> list[ModelContractError]:
        """Get merge errors from last discover_contracts call.

        Returns:
            List of ModelContractError from the last merge operation.
            Empty if discover_contracts hasn't been called or no conflicts.
        """
        return self._last_merge_errors.copy()

    def _merge_contracts(
        self,
        contracts: list[ModelDiscoveredContract],
    ) -> tuple[list[ModelDiscoveredContract], list[ModelContractError]]:
        """Merge contracts with conflict detection.

        Groups contracts by handler_id (extracted from YAML), then:
        - Single occurrence: Keep as-is
        - Multiple with same hash: Deduplicate (keep first)
        - Multiple with different hash: Generate conflict error

        Note: Contracts without handler_id (failed parsing) are kept
        as-is and will generate validation errors later.

        Args:
            contracts: List of contracts with content_hash computed

        Returns:
            Tuple of (merged_contracts, conflict_errors)
        """
        merged: list[ModelDiscoveredContract] = []
        conflicts: list[ModelContractError] = []

        # Track seen handler_ids with their hash
        seen: dict[
            str, tuple[ModelDiscoveredContract, str]
        ] = {}  # handler_id -> (contract, hash)

        for contract in contracts:
            handler_id = contract.handler_id
            content_hash = contract.content_hash or ""

            # If no handler_id, keep contract (will fail validation later)
            if not handler_id:
                merged.append(contract)
                continue

            if handler_id not in seen:
                # First occurrence
                seen[handler_id] = (contract, content_hash)
                merged.append(contract)
            else:
                # Duplicate handler_id
                existing_contract, existing_hash = seen[handler_id]

                if content_hash == existing_hash:
                    # Same hash - dedup silently
                    logger.debug(
                        "Deduplicating contract %s (same hash): %s vs %s",
                        handler_id,
                        contract.ref,
                        existing_contract.ref,
                    )
                else:
                    # Different hash - conflict error
                    error = ModelContractError(
                        contract_path=str(contract.ref),
                        handler_id=handler_id,
                        error_type="duplicate_conflict",
                        message=(
                            f"Duplicate handler_id '{handler_id}' with different content: "
                            f"found in {contract.origin}:{contract.ref} "
                            f"(hash: {content_hash[:8]}...) but already exists in "
                            f"{existing_contract.origin}:{existing_contract.ref} "
                            f"(hash: {existing_hash[:8]}...)"
                        ),
                    )
                    conflicts.append(error)

                    logger.warning(
                        "Conflict detected for handler_id %s: %s vs %s",
                        handler_id,
                        contract.ref,
                        existing_contract.ref,
                    )

        return merged, conflicts


__all__ = ["SourceContractComposite"]
