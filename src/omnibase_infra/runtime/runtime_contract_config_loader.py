# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unified loader for runtime contract configuration.

The RuntimeContractConfigLoader class that scans
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

import yaml

from omnibase_core.models.contracts.model_contract_base import ModelContractBase
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.bindings import ModelOperationBindingsSubcontract
from omnibase_infra.models.routing import ModelRoutingSubcontract
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

    Example:
        >>> loader = RuntimeContractConfigLoader()
        >>> config = loader.load_all_contracts(
        ...     search_paths=[Path("src/nodes")],
        ... )
        >>> if config.all_successful:
        ...     print("All contracts loaded successfully")

    Note:
        Namespace allowlisting for handler imports is configured at the
        HandlerPluginLoader layer, not at this config loading layer.
        See CLAUDE.md Handler Plugin Loader security patterns.
    """

    def __init__(
        self,
        scan_exclude_patterns: tuple[str, ...] = (),
        scan_deny_paths: tuple[str, ...] = (),
    ) -> None:
        """Initialize the contract config loader.

        Args:
            scan_exclude_patterns: Glob patterns for paths to exclude from scanning.
                Loaded from ``contract_loader_effect.yaml`` via
                ``ModelRuntimeNodeGraphConfig.scan_exclude_patterns``.
            scan_deny_paths: Path prefixes that are denied for security reasons.
                Loaded from ``contract_loader_effect.yaml`` via
                ``ModelRuntimeNodeGraphConfig.scan_deny_paths``.
        """
        self._scan_exclude_patterns = scan_exclude_patterns
        self._scan_deny_paths = scan_deny_paths

    def validate_path(self, path: Path) -> None:
        """Validate that a path is not in the deny list.

        Resolves the real path first to handle macOS symlinks (e.g.,
        ``/var`` → ``/private/var``). Deny patterns are matched as path
        prefixes against the resolved path, ensuring that
        ``/private/var/folders/...`` (macOS temp dirs) is not falsely
        denied by the ``/var`` pattern.

        Args:
            path: Path to validate.

        Raises:
            ProtocolConfigurationError: If the path matches a deny pattern.
        """
        original_str = str(path)
        resolved = path.resolve()
        resolved_str = str(resolved)
        for deny in self._scan_deny_paths:
            # Resolve the deny pattern too so macOS symlinks
            # (e.g. /etc → /private/etc) are handled consistently.
            resolved_deny = str(Path(deny).resolve()) if deny.startswith("/") else deny
            if (
                resolved_str.startswith(resolved_deny + "/")
                or resolved_str == resolved_deny
                or deny in original_str.split("/")
            ):
                context = ModelInfraErrorContext.with_correlation(
                    operation="validate_scan_path",
                    target_name=resolved_str,
                )
                raise ProtocolConfigurationError(
                    f"Path denied by contract security policy: {resolved_str} "
                    f"(matched deny pattern: {deny})",
                    context=context,
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
            # Validate search path against deny list
            self.validate_path(search_path)

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

            # Filter out excluded patterns and denied paths
            filtered: list[Path] = []
            for p in found:
                # Check deny paths (resolve symlinks for macOS /var → /private/var)
                resolved_str = str(p.resolve())
                denied = False
                for deny in self._scan_deny_paths:
                    resolved_deny = (
                        str(Path(deny).resolve()) if deny.startswith("/") else deny
                    )
                    if (
                        resolved_deny in str(p).split("/")
                        or resolved_str.startswith(resolved_deny + "/")
                        or resolved_str == resolved_deny
                    ):
                        logger.debug("Denied path skipped: %s (pattern: %s)", p, deny)
                        denied = True
                        break
                if denied:
                    continue

                # Check exclude patterns
                excluded = False
                for pattern in self._scan_exclude_patterns:
                    if p.match(pattern):
                        logger.debug(
                            "Excluded path skipped: %s (pattern: %s)", p, pattern
                        )
                        excluded = True
                        break
                if excluded:
                    continue

                filtered.append(p)

            logger.debug(
                "After filtering: %d of %d contract.yaml files in %s",
                len(filtered),
                len(found),
                search_path,
            )
            contract_paths.extend(filtered)

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

        Note:
            Empty operation_bindings sections (present in YAML but containing
            no bindings or global_bindings) are intentionally treated as "not
            present" and result in operation_bindings=None. This simplifies
            downstream consumers who only need to check for None rather than
            also checking for empty collections. Callers cannot distinguish
            between "section missing from YAML" and "section present but empty"
            - both result in operation_bindings=None.
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
        except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
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
                # NOTE: Empty operation_bindings sections (present but no bindings) are
                # intentionally converted to None. This simplifies downstream consumers
                # who only care about actionable configuration. Callers cannot distinguish
                # "missing section" from "empty section" - both result in None.
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
        except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
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

    def load_from_directory(self, search_path: Path) -> list[ModelContractBase]:
        """Scan a directory tree for contract.yaml files and return typed contract models.

        Invokes model_validate() on each discovered YAML, dispatching to the
        appropriate concrete ModelContract subclass based on node_type. Invalid
        or unparseable contracts are logged and skipped.

        Args:
            search_path: Root directory to scan recursively for contract.yaml files.

        Returns:
            list[ModelContractBase] — one entry per successfully validated contract.
        """
        from omnibase_core.models.contracts.model_contract_compute import (
            ModelContractCompute,
        )
        from omnibase_core.models.contracts.model_contract_effect import (
            ModelContractEffect,
        )
        from omnibase_core.models.contracts.model_contract_orchestrator import (
            ModelContractOrchestrator,
        )
        from omnibase_core.models.contracts.model_contract_reducer import (
            ModelContractReducer,
        )

        contract_paths = self._scan_for_contracts([search_path])
        results: list[ModelContractBase] = []

        for path in contract_paths:
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
            except Exception as e:  # noqa: BLE001 — boundary: log and skip malformed YAML
                logger.warning("Skipping %s — YAML parse error: %s", path, e)
                continue

            if not isinstance(raw, dict):
                logger.warning("Skipping %s — contract root is not a mapping", path)
                continue

            node_type = str(raw.get("node_type", "")).upper()
            contract_class: type[ModelContractBase]
            if "ORCHESTRATOR" in node_type:
                contract_class = ModelContractOrchestrator
            elif "REDUCER" in node_type:
                contract_class = ModelContractReducer
            elif "COMPUTE" in node_type:
                contract_class = ModelContractCompute
            else:
                contract_class = ModelContractEffect

            model_fields = set(contract_class.model_fields.keys())
            filtered = {k: v for k, v in raw.items() if k in model_fields}

            try:
                results.append(contract_class.model_validate(filtered))
            except Exception as e:  # noqa: BLE001 — boundary: log and skip invalid contracts
                logger.warning(
                    "Skipping %s — model_validate() failed as %s: %s",
                    path,
                    contract_class.__name__,
                    e,
                )

        return results

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
