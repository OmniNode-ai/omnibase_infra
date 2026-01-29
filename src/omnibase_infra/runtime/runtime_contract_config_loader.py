# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unified loader for runtime contract configuration.

This module provides the RuntimeContractConfigLoader class that scans
directories for contract.yaml files at startup and loads all subcontracts
(handler_routing, operation_bindings) into a consolidated configuration.

Part of OMN-1519: Runtime contract config loader.

Design Pattern:
    RuntimeContractConfigLoader acts as an orchestrator for the individual
    subcontract loaders (handler_routing_loader, operation_bindings_loader).
    It scans directories, delegates loading to specialized loaders, and
    aggregates results into a single ModelRuntimeContractConfig.

Error Handling:
    The loader uses a graceful error handling strategy - individual contract
    failures are logged and collected but do not stop the loading of other
    contracts. This ensures the runtime can start even if some contracts
    are malformed.

Thread Safety:
    RuntimeContractConfigLoader is designed for single-threaded use during
    startup. The resulting ModelRuntimeContractConfig is immutable and
    thread-safe for concurrent read access.

Example:
    >>> from pathlib import Path
    >>> from omnibase_infra.runtime import RuntimeContractConfigLoader
    >>>
    >>> # Uses TRUSTED_HANDLER_NAMESPACE_PREFIXES by default
    >>> loader = RuntimeContractConfigLoader()
    >>> config = loader.load_all_contracts(
    ...     search_paths=[Path("src/omnibase_infra/nodes")],
    ... )
    >>> print(f"Loaded {config.total_contracts_loaded} contracts")
    >>> for path, routing in config.handler_routing_configs.items():
    ...     print(f"  {path}: {len(routing.handlers)} handlers")

.. versionadded:: 0.2.8
    Created as part of OMN-1519.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.bindings import ModelOperationBindingsSubcontract
from omnibase_infra.models.routing import ModelRoutingSubcontract
from omnibase_infra.runtime.constants_security import (
    TRUSTED_HANDLER_NAMESPACE_PREFIXES,
)
from omnibase_infra.runtime.contract_loaders.handler_routing_loader import (
    load_handler_routing_subcontract,
)
from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
    load_operation_bindings_subcontract,
)
from omnibase_infra.runtime.models.model_contract_load_result import (
    ModelContractLoadResult,
)
from omnibase_infra.runtime.models.model_runtime_contract_config import (
    ModelRuntimeContractConfig,
)

logger = logging.getLogger(__name__)

# Contract file name to search for
CONTRACT_YAML_FILENAME: str = "contract.yaml"


class RuntimeContractConfigLoader:
    """Unified loader for runtime contract configuration.

    Scans directories for contract.yaml files and loads all subcontracts
    (handler_routing, operation_bindings) at startup. Individual loaders
    are delegated to for specific subcontract types.

    Attributes:
        allowed_namespaces: Optional list of allowed namespace prefixes for
            handler module imports. If set, handler modules outside these
            namespaces will be rejected for security.

    Example:
        >>> # Default: uses TRUSTED_HANDLER_NAMESPACE_PREFIXES
        >>> loader = RuntimeContractConfigLoader()
        >>> config = loader.load_all_contracts(
        ...     search_paths=[Path("src/nodes")],
        ... )
        >>> if config.all_successful:
        ...     print("All contracts loaded successfully")

    Note:
        The allowed_namespaces parameter is passed through to the
        handler routing loader for security validation. See CLAUDE.md
        Handler Plugin Loader security patterns.
    """

    def __init__(self, allowed_namespaces: list[str] | None = None) -> None:
        """Initialize loader with optional namespace restrictions.

        Args:
            allowed_namespaces: Optional list of allowed namespace prefixes
                for handler module imports. If None, defaults to
                TRUSTED_HANDLER_NAMESPACE_PREFIXES (omnibase_core., omnibase_infra.).
                Pass an explicit list to extend or override the defaults.

        Security:
            The default namespaces form a security boundary. Only extend
            this list via explicit security configuration, not env vars.
            See constants_security.py for rationale.
        """
        self._allowed_namespaces = (
            allowed_namespaces
            if allowed_namespaces is not None
            else list(TRUSTED_HANDLER_NAMESPACE_PREFIXES)
        )

    def load_all_contracts(
        self,
        search_paths: list[Path],
        correlation_id: UUID | None = None,
    ) -> ModelRuntimeContractConfig:
        """Scan directories and load all contract.yaml files.

        Recursively scans the provided directories for contract.yaml files,
        loading handler_routing and operation_bindings subcontracts from each.
        Errors are collected per-contract without stopping the overall load.

        Args:
            search_paths: Directories to scan for contract.yaml files.
            correlation_id: Optional correlation ID for tracing. If not
                provided, a new UUID is generated.

        Returns:
            ModelRuntimeContractConfig containing consolidated configuration
            from all loaded contracts, including any errors encountered.

        Example:
            >>> config = loader.load_all_contracts(
            ...     search_paths=[
            ...         Path("src/omnibase_infra/nodes"),
            ...         Path("src/myapp/nodes"),
            ...     ],
            ... )
            >>> print(f"Success rate: {config.success_rate:.1%}")
        """
        correlation_id = correlation_id or uuid4()

        logger.info(
            "Starting contract config load with correlation_id=%s, search_paths=%s",
            correlation_id,
            [str(p) for p in search_paths],
        )

        # Find all contract.yaml files
        contract_paths = self._scan_for_contracts(search_paths)
        total_found = len(contract_paths)

        logger.info(
            "Found %d contract.yaml files to load (correlation_id=%s)",
            total_found,
            correlation_id,
        )

        # Load each contract
        results: list[ModelContractLoadResult] = []
        for contract_path in contract_paths:
            result = self._load_single_contract(contract_path, correlation_id)
            results.append(result)

        # Calculate totals
        total_loaded = sum(1 for r in results if r.success)
        total_errors = sum(1 for r in results if not r.success)

        # Log summary
        if total_errors > 0:
            logger.warning(
                "Contract loading completed with errors: "
                "found=%d, loaded=%d, errors=%d (correlation_id=%s)",
                total_found,
                total_loaded,
                total_errors,
                correlation_id,
            )
            for result in results:
                if not result.success:
                    logger.warning(
                        "  Failed: %s - %s",
                        result.contract_path,
                        result.error,
                    )
        else:
            logger.info(
                "Contract loading completed successfully: "
                "found=%d, loaded=%d (correlation_id=%s)",
                total_found,
                total_loaded,
                correlation_id,
            )

        return ModelRuntimeContractConfig(
            contract_results=results,
            total_contracts_found=total_found,
            total_contracts_loaded=total_loaded,
            total_errors=total_errors,
            correlation_id=correlation_id,
        )

    def _scan_for_contracts(self, search_paths: list[Path]) -> list[Path]:
        """Find all contract.yaml files in search paths.

        Recursively scans each search path for contract.yaml files.
        Paths that don't exist are logged as warnings and skipped.

        Args:
            search_paths: Directories to scan.

        Returns:
            List of paths to contract.yaml files, sorted by path for
            deterministic ordering.
        """
        contract_paths: list[Path] = []

        for search_path in search_paths:
            if not search_path.exists():
                logger.warning(
                    "Search path does not exist, skipping: %s",
                    search_path,
                )
                continue

            if not search_path.is_dir():
                logger.warning(
                    "Search path is not a directory, skipping: %s",
                    search_path,
                )
                continue

            # Use glob to find all contract.yaml files recursively
            found = list(search_path.glob(f"**/{CONTRACT_YAML_FILENAME}"))
            logger.debug(
                "Found %d contract.yaml files in %s",
                len(found),
                search_path,
            )
            contract_paths.extend(found)

        # Sort for deterministic ordering
        return sorted(contract_paths)

    def _load_single_contract(
        self,
        contract_path: Path,
        correlation_id: UUID,
    ) -> ModelContractLoadResult:
        """Load handler_routing and operation_bindings from a single contract.

        Attempts to load both subcontracts from the contract.yaml file.
        Either or both may be missing - only present sections are loaded.
        Errors from either loader are caught and reported.

        Args:
            contract_path: Path to the contract.yaml file.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelContractLoadResult with loaded subcontracts or error.
        """
        logger.debug(
            "Loading contract: %s (correlation_id=%s)",
            contract_path,
            correlation_id,
        )

        handler_routing: ModelRoutingSubcontract | None = None
        operation_bindings: ModelOperationBindingsSubcontract | None = None
        errors: list[str] = []

        # Try to load handler_routing
        try:
            handler_routing = load_handler_routing_subcontract(contract_path)
            logger.debug(
                "Loaded handler_routing from %s: %d handlers",
                contract_path,
                len(handler_routing.handlers),
            )
        except ProtocolConfigurationError as e:
            # Check if this is "missing section" which is OK
            if (
                "MISSING_HANDLER_ROUTING" in str(e)
                or "handler_routing" in str(e).lower()
            ):
                logger.debug(
                    "No handler_routing section in %s (this is OK)",
                    contract_path,
                )
            else:
                error_msg = f"handler_routing load failed: {e}"
                logger.warning(
                    "Failed to load handler_routing from %s: %s",
                    contract_path,
                    e,
                )
                errors.append(error_msg)
        except Exception as e:
            error_msg = f"handler_routing load failed: {type(e).__name__}: {e}"
            logger.warning(
                "Unexpected error loading handler_routing from %s: %s",
                contract_path,
                e,
            )
            errors.append(error_msg)

        # Try to load operation_bindings
        try:
            operation_bindings = load_operation_bindings_subcontract(contract_path)
            if operation_bindings.bindings or operation_bindings.global_bindings:
                logger.debug(
                    "Loaded operation_bindings from %s: %d operations, %d global bindings",
                    contract_path,
                    len(operation_bindings.bindings),
                    len(operation_bindings.global_bindings or []),
                )
            else:
                # Empty bindings - treat as not present
                logger.debug(
                    "No operation_bindings content in %s (empty section)",
                    contract_path,
                )
                operation_bindings = None
        except ProtocolConfigurationError as e:
            # Check if this is "missing section" or "file not found" which is OK
            if "CONTRACT_NOT_FOUND" in str(e) or "operation_bindings" in str(e).lower():
                logger.debug(
                    "No operation_bindings section in %s (this is OK)",
                    contract_path,
                )
            else:
                error_msg = f"operation_bindings load failed: {e}"
                logger.warning(
                    "Failed to load operation_bindings from %s: %s",
                    contract_path,
                    e,
                )
                errors.append(error_msg)
        except Exception as e:
            error_msg = f"operation_bindings load failed: {type(e).__name__}: {e}"
            logger.warning(
                "Unexpected error loading operation_bindings from %s: %s",
                contract_path,
                e,
            )
            errors.append(error_msg)

        # Build result
        if errors:
            combined_error = "; ".join(errors)
            return ModelContractLoadResult.failed(
                contract_path=contract_path,
                error=combined_error,
                correlation_id=correlation_id,
            )

        return ModelContractLoadResult.succeeded(
            contract_path=contract_path,
            handler_routing=handler_routing,
            operation_bindings=operation_bindings,
            correlation_id=correlation_id,
        )

    def load_single_contract(
        self,
        contract_path: Path,
        correlation_id: UUID | None = None,
    ) -> ModelContractLoadResult:
        """Load a single contract.yaml file (public API).

        Convenience method for loading a single contract without scanning.
        Useful for testing or targeted loading.

        Args:
            contract_path: Path to the contract.yaml file.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelContractLoadResult with loaded subcontracts or error.

        Raises:
            ProtocolConfigurationError: If contract_path does not exist.
        """
        correlation_id = correlation_id or uuid4()

        if not contract_path.exists():
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_single_contract",
                target_name=str(contract_path),
            )
            raise ProtocolConfigurationError(
                f"Contract file not found: {contract_path}",
                context=ctx,
            )

        return self._load_single_contract(contract_path, correlation_id)


__all__ = [
    "CONTRACT_YAML_FILENAME",
    "RuntimeContractConfigLoader",
]
