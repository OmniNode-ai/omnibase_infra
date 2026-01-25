# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Contract Source for Filesystem Discovery.

This module provides HandlerContractSource, which discovers handler contracts
from the filesystem by recursively scanning configured paths for
handler_contract.yaml files, parsing them, and transforming them into
ModelHandlerDescriptor instances wrapped in ModelContractDiscoveryResult.

Part of OMN-1097: HandlerContractSource + Filesystem Discovery.

The source implements ProtocolContractSource and supports two operation modes:
- Strict mode (default): Raises on first error encountered
- Graceful mode: Collects errors, continues discovery

Both modes return ModelContractDiscoveryResult for a consistent interface.
In strict mode, validation_errors will be empty since errors raise exceptions.

See Also:
    - ProtocolContractSource: Protocol definition for handler sources
    - ModelHandlerContract: Contract model from omnibase_core
    - ModelHandlerValidationError: Structured error model for validation failures
    - ModelContractDiscoveryResult: Result model containing descriptors and errors

.. versionadded:: 0.6.2
    Created as part of OMN-1097 filesystem handler discovery.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import cast

import yaml
from pydantic import ValidationError

from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import (
    LiteralHandlerKind,
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
    ModelHandlerIdentifier,
)
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource

# Forward Reference Resolution:
# ModelContractDiscoveryResult uses a forward reference to ModelHandlerValidationError.
# Since we import ModelHandlerValidationError above, we can call model_rebuild() here
# to resolve the forward reference. This call is idempotent - multiple calls are harmless.
# This ensures the model is fully defined before we create instances in discover_handlers().
ModelContractDiscoveryResult.model_rebuild()

logger = logging.getLogger(__name__)

# File pattern for handler contracts
HANDLER_CONTRACT_FILENAME = "handler_contract.yaml"


# Maximum contract file size (10MB) to prevent memory exhaustion
MAX_CONTRACT_SIZE = 10 * 1024 * 1024


# =============================================================================
# Module-Level Helper Functions
# =============================================================================
#
# These functions are extracted from HandlerContractSource to reduce method count
# while maintaining the same functionality. They are pure functions that operate
# on their inputs without requiring instance state.
# =============================================================================


def _sanitize_path_for_logging(path: Path) -> str:
    """Sanitize a file path for safe inclusion in logs and error messages.

    In production environments, full paths may leak sensitive information
    about directory structure. This function returns only the filename and
    parent directory to provide context without exposing full paths.

    Args:
        path: The full path to sanitize.

    Returns:
        Sanitized path string showing only parent/filename.
        For example: "/home/user/code/handlers/handler_contract.yaml"
        becomes "handlers/handler_contract.yaml".
    """
    # Return parent directory name + filename for context
    # This provides enough info for debugging without full path exposure
    try:
        return str(Path(path.parent.name) / path.name)
    except (ValueError, AttributeError):
        # Fallback to just filename if parent extraction fails
        return path.name


def _create_parse_error(
    contract_path: Path,
    error: yaml.YAMLError,
) -> ModelHandlerValidationError:
    """Create a validation error for YAML parse failures.

    Args:
        contract_path: Path to the failing contract file.
        error: The YAML parsing error.

    Returns:
        ModelHandlerValidationError with parse error details.
    """
    handler_identity = ModelHandlerIdentifier.from_handler_id(
        f"unknown@{contract_path.name}"
    )

    return ModelHandlerValidationError(
        error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
        rule_id="CONTRACT-001",
        handler_identity=handler_identity,
        source_type=EnumHandlerSourceType.CONTRACT,
        message=f"Failed to parse YAML in {_sanitize_path_for_logging(contract_path)}: {error}",
        remediation_hint="Check YAML syntax and ensure proper indentation",
        file_path=str(contract_path),
    )


def _create_validation_error(
    contract_path: Path,
    error: ValidationError,
) -> ModelHandlerValidationError:
    """Create a validation error for contract validation failures.

    Args:
        contract_path: Path to the failing contract file.
        error: The Pydantic validation error.

    Returns:
        ModelHandlerValidationError with validation details.
    """
    handler_identity = ModelHandlerIdentifier.from_handler_id(
        f"unknown@{contract_path.name}"
    )

    # Extract first error detail for remediation hint
    error_details = error.errors()
    if error_details:
        first_error = error_details[0]
        field_loc = " -> ".join(str(x) for x in first_error.get("loc", ()))
        error_msg = str(first_error.get("msg", "validation failed"))
    else:
        field_loc = "unknown"
        error_msg = "validation failed"

    return ModelHandlerValidationError(
        error_type=EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR,
        rule_id="CONTRACT-002",
        handler_identity=handler_identity,
        source_type=EnumHandlerSourceType.CONTRACT,
        message=f"Contract validation failed in {_sanitize_path_for_logging(contract_path)}: {error_msg} at {field_loc}",
        remediation_hint=f"Check the '{field_loc}' field in the contract",
        file_path=str(contract_path),
    )


def _create_size_limit_error(
    contract_path: Path,
    file_size: int,
) -> ModelHandlerValidationError:
    """Create a validation error for file size limit violations.

    Args:
        contract_path: Path to the oversized contract file.
        file_size: The actual file size in bytes.

    Returns:
        ModelHandlerValidationError with size limit details.
    """
    handler_identity = ModelHandlerIdentifier.from_handler_id(
        f"unknown@{contract_path.name}"
    )

    return ModelHandlerValidationError(
        error_type=EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR,
        rule_id="CONTRACT-003",
        handler_identity=handler_identity,
        source_type=EnumHandlerSourceType.CONTRACT,
        message=(
            f"Contract file {_sanitize_path_for_logging(contract_path)} exceeds size limit: "
            f"{file_size} bytes (max: {MAX_CONTRACT_SIZE} bytes)"
        ),
        remediation_hint=(
            f"Reduce contract file size to under {MAX_CONTRACT_SIZE // (1024 * 1024)}MB. "
            "Consider splitting into multiple contracts if needed."
        ),
        file_path=str(contract_path),
    )


def _create_io_error(
    contract_path: Path,
    error: OSError,
) -> ModelHandlerValidationError:
    """Create a validation error for I/O failures.

    Args:
        contract_path: Path to the contract file that failed to read.
        error: The I/O error encountered.

    Returns:
        ModelHandlerValidationError with I/O error details.
    """
    handler_identity = ModelHandlerIdentifier.from_handler_id(
        f"unknown@{contract_path.name}"
    )

    # OSError.strerror may be None for some error types (e.g., custom subclasses),
    # so use str(error) as a fallback to ensure we always have an error message
    error_message = error.strerror or str(error)

    return ModelHandlerValidationError(
        error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
        rule_id="CONTRACT-004",
        handler_identity=handler_identity,
        source_type=EnumHandlerSourceType.CONTRACT,
        message=f"Failed to read contract file: {error_message}",
        remediation_hint="Check file permissions and ensure the file exists",
        file_path=str(contract_path),
    )


def _create_version_parse_error(
    contract_path: Path,
    error_message: str,
) -> ModelHandlerValidationError:
    """Create a validation error for version string parse failures.

    Args:
        contract_path: Path to the contract file with invalid version.
        error_message: The error message describing the version parse failure.

    Returns:
        ModelHandlerValidationError with version parse error details.
    """
    handler_identity = ModelHandlerIdentifier.from_handler_id(
        f"unknown@{contract_path.name}"
    )

    return ModelHandlerValidationError(
        error_type=EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR,
        rule_id="CONTRACT-005",
        handler_identity=handler_identity,
        source_type=EnumHandlerSourceType.CONTRACT,
        message=(
            f"Invalid version string in contract "
            f"{_sanitize_path_for_logging(contract_path)}: {error_message}"
        ),
        remediation_hint=(
            "Ensure the 'version' field uses semantic versioning format "
            "(e.g., '1.0.0', '2.1.3-beta.1'). "
            "Version components must be non-negative integers."
        ),
        file_path=str(contract_path),
    )


# =============================================================================
# HandlerContractSource Implementation
# =============================================================================


class HandlerContractSource(ProtocolContractSource):
    """Handler source that discovers contracts from the filesystem.

    This class implements ProtocolContractSource by recursively scanning
    configured paths for handler_contract.yaml files, parsing them with
    YAML and validating against ModelHandlerContract from omnibase_core.

    Protocol Compliance:
        This class explicitly inherits from ProtocolContractSource and implements
        all required protocol methods: discover_handlers() async method and
        source_type property. Protocol compliance is verified at runtime through
        Python's structural subtyping and enforced by type checkers (mypy/pyright).

    The source supports two operation modes:
    - Strict mode (default): Raises ModelOnexError on first error
    - Graceful mode: Collects all errors, continues discovery

    Both modes return ModelContractDiscoveryResult for a consistent interface.
    In strict mode, validation_errors will always be empty since errors raise
    exceptions instead of being collected.

    Attributes:
        source_type: Returns "CONTRACT" as the source type identifier.

    Example:
        >>> # Strict mode discovery (raises on error)
        >>> source = HandlerContractSource(contract_paths=[Path("./handlers")])
        >>> result = await source.discover_handlers()
        >>> print(f"Found {len(result.descriptors)} handlers")
        >>> # result.validation_errors is always empty in strict mode

        >>> # Graceful mode with error collection
        >>> source = HandlerContractSource(
        ...     contract_paths=[Path("./handlers")],
        ...     graceful_mode=True,
        ... )
        >>> result = await source.discover_handlers()
        >>> print(f"Found {len(result.descriptors)} handlers")
        >>> print(f"Encountered {len(result.validation_errors)} errors")

    Performance Characteristics:
        - File system scanning is O(n) where n is total files in paths
        - YAML parsing is synchronous (consider aiofiles for high-throughput)
        - Typical performance: 100-500 contracts/second on SSD
        - Memory: ~1KB per contract descriptor retained

    .. versionadded:: 0.6.2
        Created as part of OMN-1097 filesystem handler discovery.
    """

    def __init__(
        self,
        contract_paths: list[Path],
        graceful_mode: bool = False,
    ) -> None:
        """Initialize the handler contract source.

        Args:
            contract_paths: List of paths to scan for handler_contract.yaml files.
                Must not be empty.
            graceful_mode: If True, collect errors and continue discovery.
                If False (default), raise on first error.

        Raises:
            ModelOnexError: If contract_paths is empty.
        """
        if not contract_paths:
            raise ModelOnexError(
                "contract_paths is required and cannot be empty",
                error_code="HANDLER_SOURCE_001",
            )

        self._contract_paths = contract_paths
        self._graceful_mode = graceful_mode

    @property
    def source_type(self) -> str:
        """Return the source type identifier.

        Returns:
            "CONTRACT" as the source type.
        """
        return "CONTRACT"

    async def discover_handlers(
        self,
    ) -> ModelContractDiscoveryResult:
        """Discover handler contracts from configured paths.

        Recursively scans all configured paths for handler_contract.yaml files,
        parses them, validates against ModelHandlerContract, and transforms
        them into ModelHandlerDescriptor instances.

        In strict mode (default), raises on the first error encountered.
        In graceful mode, collects all errors and continues discovery.

        Returns:
            ModelContractDiscoveryResult containing discovered descriptors and
            any validation errors. In strict mode, validation_errors will be
            empty (errors raise exceptions instead of being collected).

        Raises:
            ModelOnexError: In strict mode, if a path doesn't exist or
                a contract fails to parse/validate.

        Performance:
            Discovery is synchronous and scales linearly with the number
            of files. Telemetry is logged including duration_seconds and
            contracts_per_second for monitoring.
        """
        start_time = time.perf_counter()
        descriptors: list[ModelHandlerDescriptor] = []
        validation_errors: list[ModelHandlerValidationError] = []
        # Track discovered files to avoid duplicates when paths overlap
        discovered_paths: set[Path] = set()

        logger.debug(
            "Starting handler contract discovery",
            extra={
                "paths_scanned": len(self._contract_paths),
                "graceful_mode": self._graceful_mode,
                "contract_paths": [str(p) for p in self._contract_paths],
            },
        )

        for base_path in self._contract_paths:
            # Check if path exists (strict mode raises, graceful collects)
            if not base_path.exists():
                error_msg = f"Contract path does not exist: {base_path}"
                if not self._graceful_mode:
                    raise ModelOnexError(
                        error_msg,
                        error_code="HANDLER_SOURCE_002",
                    )
                # In graceful mode, log and continue
                logger.warning(
                    "Contract path does not exist, skipping: %s",
                    base_path,
                    extra={
                        "path": str(base_path),
                        "graceful_mode": self._graceful_mode,
                        "paths_scanned": len(self._contract_paths),
                    },
                )
                continue

            # Discover contract files
            contract_files = self._find_contract_files(base_path)
            logger.debug(
                "Scanned path for contracts: %s",
                base_path,
                extra={
                    "base_path": str(base_path),
                    "contracts_found": len(contract_files),
                    "graceful_mode": self._graceful_mode,
                    "paths_scanned": len(self._contract_paths),
                },
            )

            for contract_file in contract_files:
                # Deduplicate using resolved path to handle overlapping search paths
                resolved_path = contract_file.resolve()
                if resolved_path in discovered_paths:
                    continue

                # Symlink protection: verify resolved path is within configured paths
                # This prevents symlink-based path traversal attacks where a symlink
                # inside a configured path points to files outside allowed directories
                is_within_allowed = any(
                    resolved_path.is_relative_to(base.resolve())
                    for base in self._contract_paths
                )
                if not is_within_allowed:
                    logger.warning(
                        "Skipping contract file outside allowed paths: %s (resolved to %s)",
                        contract_file,
                        resolved_path,
                        extra={
                            "contract_file": str(contract_file),
                            "resolved_path": str(resolved_path),
                            "graceful_mode": self._graceful_mode,
                            "reason": "symlink_outside_allowed_paths",
                        },
                    )
                    continue

                discovered_paths.add(resolved_path)

                try:
                    descriptor = self._parse_contract_file(contract_file)
                    descriptors.append(descriptor)
                    logger.debug(
                        "Successfully parsed contract: %s",
                        contract_file,
                        extra={
                            "contract_file": str(contract_file),
                            "handler_id": descriptor.handler_id,
                            "handler_name": descriptor.name,
                            "handler_version": descriptor.version,
                            "graceful_mode": self._graceful_mode,
                        },
                    )
                except yaml.YAMLError as e:
                    error = _create_parse_error(contract_file, e)
                    if not self._graceful_mode:
                        raise ModelOnexError(
                            f"Failed to parse YAML contract at {contract_file}: {e}",
                            error_code="HANDLER_SOURCE_003",
                        ) from e
                    logger.warning(
                        "Failed to parse YAML contract in %s, continuing in graceful mode",
                        _sanitize_path_for_logging(contract_file),
                        extra={
                            "contract_file": str(contract_file),
                            "error_type": "yaml_parse_error",
                            "graceful_mode": self._graceful_mode,
                            "paths_scanned": len(self._contract_paths),
                        },
                    )
                    validation_errors.append(error)
                except ValidationError as e:
                    error = _create_validation_error(contract_file, e)
                    if not self._graceful_mode:
                        raise ModelOnexError(
                            f"Contract validation failed at {contract_file}: {e}",
                            error_code="HANDLER_SOURCE_004",
                        ) from e
                    logger.warning(
                        "Contract validation failed in %s, continuing in graceful mode",
                        _sanitize_path_for_logging(contract_file),
                        extra={
                            "contract_file": str(contract_file),
                            "error_type": "validation_error",
                            "error_count": len(e.errors()),
                            "graceful_mode": self._graceful_mode,
                            "paths_scanned": len(self._contract_paths),
                        },
                    )
                    validation_errors.append(error)
                except ModelOnexError as e:
                    # Handle known ModelOnexError types in graceful mode
                    if not self._graceful_mode:
                        raise

                    # Handle specific error codes gracefully:
                    # - HANDLER_SOURCE_005: File size limit exceeded
                    # - HANDLER_SOURCE_007: Invalid version string
                    # Other ModelOnexError types should be re-raised as they may indicate
                    # more serious issues (e.g., configuration errors, programming errors)
                    # Defensive check: error_code should always exist on ModelOnexError,
                    # but handle the case where it might be None
                    error_code = getattr(e, "error_code", None)
                    if error_code == "HANDLER_SOURCE_005":
                        # Get file size defensively - the original stat() that triggered
                        # this error may have succeeded, but the file could have changed
                        # (TOCTOU race). Use 0 as fallback if stat() fails now.
                        try:
                            file_size = contract_file.stat().st_size
                        except OSError:
                            file_size = 0  # File may have been deleted/changed
                        error = _create_size_limit_error(
                            contract_file,
                            file_size,
                        )
                        logger.warning(
                            "Contract file %s exceeds size limit, continuing in graceful mode",
                            _sanitize_path_for_logging(contract_file),
                            extra={
                                "contract_file": str(contract_file),
                                "error_type": "size_limit_error",
                                "error_code": error_code,
                                "graceful_mode": self._graceful_mode,
                                "paths_scanned": len(self._contract_paths),
                            },
                        )
                        validation_errors.append(error)
                    elif error_code == "HANDLER_SOURCE_007":
                        # Invalid version string - extract version from error message
                        error = _create_version_parse_error(
                            contract_file,
                            str(e),
                        )
                        logger.warning(
                            "Contract file %s has invalid version string, "
                            "continuing in graceful mode",
                            _sanitize_path_for_logging(contract_file),
                            extra={
                                "contract_file": str(contract_file),
                                "error_type": "version_parse_error",
                                "error_code": error_code,
                                "error_message": str(e),
                                "graceful_mode": self._graceful_mode,
                                "paths_scanned": len(self._contract_paths),
                            },
                        )
                        validation_errors.append(error)
                    else:
                        # Re-raise unexpected ModelOnexError types even in graceful mode
                        # These may indicate configuration or programming errors
                        raise
                except OSError as e:
                    # Handle file I/O errors (permission denied, file not found, etc.)
                    if not self._graceful_mode:
                        raise ModelOnexError(
                            f"Failed to read contract file at {contract_file}: {e}",
                            error_code="HANDLER_SOURCE_006",
                        ) from e
                    error = _create_io_error(contract_file, e)
                    logger.warning(
                        "Failed to read contract file, continuing in graceful mode: %s",
                        _sanitize_path_for_logging(contract_file),
                        extra={
                            "contract_file": str(contract_file),
                            "error_type": "io_error",
                            "error_message": str(e),
                            "graceful_mode": self._graceful_mode,
                        },
                    )
                    validation_errors.append(error)

        # Calculate duration and log results
        duration_seconds = time.perf_counter() - start_time
        self._log_discovery_results(
            len(descriptors), len(validation_errors), duration_seconds
        )

        return ModelContractDiscoveryResult(
            descriptors=descriptors,
            validation_errors=validation_errors,
        )

    def _find_contract_files(self, base_path: Path) -> list[Path]:
        """Find all handler_contract.yaml files under a base path.

        Args:
            base_path: Directory to scan recursively.

        Returns:
            List of paths to handler_contract.yaml files.
        """
        if base_path.is_file():
            # Exact case-sensitive match for file names
            if base_path.name == HANDLER_CONTRACT_FILENAME:
                return [base_path]
            return []

        # Use rglob and filter for exact case-sensitive match
        # This ensures we don't pick up HANDLER_CONTRACT.yaml or handler_contract.yml
        return [
            f
            for f in base_path.rglob(HANDLER_CONTRACT_FILENAME)
            if f.name == HANDLER_CONTRACT_FILENAME
        ]

    def _parse_contract_file(self, contract_path: Path) -> ModelHandlerDescriptor:
        """Parse a contract file and return a descriptor.

        Args:
            contract_path: Path to the handler_contract.yaml file.

        Returns:
            ModelHandlerDescriptor created from the contract.

        Raises:
            ModelOnexError: If contract file exceeds MAX_CONTRACT_SIZE (10MB),
                or if the version string in the contract is invalid.
            yaml.YAMLError: If YAML parsing fails.
            ValidationError: If contract validation fails.
        """
        # TODO [OMN-1352]: Replace direct file I/O with FileRegistry abstraction
        #
        # Why direct file operations are used here:
        #   RegistryFileBased (or FileRegistry) does not yet exist in omnibase_core.
        #   This is a temporary implementation that will be replaced once the
        #   registry abstraction is available, providing:
        #   - Consistent file loading across the codebase
        #   - Caching and performance optimizations
        #   - Unified error handling for file operations
        #
        # See: docs/architecture/RUNTIME_HOST_IMPLEMENTATION_PLAN.md

        # Validate file size before reading to prevent memory exhaustion
        file_size = contract_path.stat().st_size
        if file_size > MAX_CONTRACT_SIZE:
            raise ModelOnexError(
                f"Contract file exceeds size limit: {file_size} bytes "
                f"(max: {MAX_CONTRACT_SIZE} bytes)",
                error_code="HANDLER_SOURCE_005",
            )

        with contract_path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        # Validate against ModelHandlerContract
        contract = ModelHandlerContract.model_validate(raw_data)

        # TODO [OMN-1420]: Extract handler_class from ModelHandlerContract
        #
        # handler_contract.yaml files include a `handler_class` field for dynamic import
        # (e.g., "omnibase_infra.handlers.handler_consul.HandlerConsul"), but
        # ModelHandlerContract from omnibase_core does not have this field yet.
        #
        # Once ModelHandlerContract is updated to include handler_class, this code
        # should be changed from:
        #     handler_class=raw_data.get("handler_class")
        # to:
        #     handler_class=contract.handler_class
        #
        # For now, extract directly from raw YAML data to support dynamic handler loading.
        # See: https://linear.app/omninode/issue/OMN-1420
        handler_class = (
            raw_data.get("handler_class") if isinstance(raw_data, dict) else None
        )

        # Use contract_version directly - it's already a ModelSemVer from Pydantic validation
        # Transform to descriptor
        return ModelHandlerDescriptor(
            handler_id=contract.handler_id,
            name=contract.name,
            version=contract.contract_version,
            handler_kind=cast(
                "LiteralHandlerKind", contract.descriptor.node_archetype.value
            ),
            input_model=contract.input_model,
            output_model=contract.output_model,
            description=contract.description,
            handler_class=handler_class,
            contract_path=str(contract_path),
        )

    def _log_discovery_results(
        self,
        discovered_count: int,
        failure_count: int,
        duration_seconds: float,
    ) -> None:
        """Log the discovery results with structured counts and timing.

        Args:
            discovered_count: Number of successfully discovered contracts.
            failure_count: Number of validation failures.
            duration_seconds: Total discovery duration in seconds.
        """
        contracts_per_sec = (
            discovered_count / duration_seconds if duration_seconds > 0 else 0.0
        )

        logger.info(
            "Handler contract discovery completed: "
            "discovered_contract_count=%d, validation_failure_count=%d, "
            "paths_scanned=%d, graceful_mode=%s, "
            "duration_seconds=%.3f, contracts_per_second=%.1f",
            discovered_count,
            failure_count,
            len(self._contract_paths),
            self._graceful_mode,
            duration_seconds,
            contracts_per_sec,
            extra={
                "discovered_contract_count": discovered_count,
                "validation_failure_count": failure_count,
                "paths_scanned": len(self._contract_paths),
                "graceful_mode": self._graceful_mode,
                "contract_paths": [str(p) for p in self._contract_paths],
                "duration_seconds": duration_seconds,
                "contracts_per_second": contracts_per_sec,
            },
        )


__all__ = [
    "HandlerContractSource",
    "MAX_CONTRACT_SIZE",
    "ModelContractDiscoveryResult",
    "ModelHandlerDescriptor",
]
