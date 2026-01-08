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
from pathlib import Path

import yaml
from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import (
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
    ModelHandlerIdentifier,
)
from omnibase_infra.runtime.protocol_contract_descriptor import (
    ProtocolContractDescriptor,
)
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource

# Rebuild ModelContractDiscoveryResult to resolve the forward reference
# to ModelHandlerValidationError. This must happen after ModelHandlerValidationError
# is imported to make the type available for Pydantic validation.
#
# Why this pattern is used:
#   ModelContractDiscoveryResult has a field typed as list[ModelHandlerValidationError].
#   ModelHandlerValidationError imports ModelHandlerIdentifier from models.handlers.
#   If ModelContractDiscoveryResult directly imported ModelHandlerValidationError,
#   it would cause a circular import because models.handlers.__init__.py imports
#   ModelContractDiscoveryResult.
#
# The solution:
#   1. ModelContractDiscoveryResult uses TYPE_CHECKING to defer the import
#   2. With PEP 563 (from __future__ import annotations), the annotation becomes
#      a string at runtime, avoiding the circular import
#   3. model_rebuild() resolves the string annotation to the actual type after
#      both classes are defined
#
# This is tested in: tests/unit/runtime/test_handler_contract_source.py
ModelContractDiscoveryResult.model_rebuild()

logger = logging.getLogger(__name__)

# File pattern for handler contracts
HANDLER_CONTRACT_FILENAME = "handler_contract.yaml"

# Maximum contract file size (10MB) to prevent memory exhaustion
MAX_CONTRACT_SIZE = 10 * 1024 * 1024


# =============================================================================
# HandlerContractSource Implementation
# =============================================================================


class HandlerContractSource:
    """Handler source that discovers contracts from the filesystem.

    This class implements ProtocolContractSource by recursively scanning
    configured paths for handler_contract.yaml files, parsing them with
    YAML and validating against ModelHandlerContract from omnibase_core.

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
        """
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
                    error = self._create_parse_error(contract_file, e)
                    if not self._graceful_mode:
                        raise ModelOnexError(
                            f"Failed to parse YAML contract at {contract_file}: {e}",
                            error_code="HANDLER_SOURCE_003",
                        ) from e
                    logger.warning(
                        "Failed to parse YAML contract, continuing in graceful mode: %s",
                        contract_file,
                        extra={
                            "contract_file": str(contract_file),
                            "error_type": "yaml_parse_error",
                            "graceful_mode": self._graceful_mode,
                            "paths_scanned": len(self._contract_paths),
                        },
                    )
                    validation_errors.append(error)
                except ValidationError as e:
                    error = self._create_validation_error(contract_file, e)
                    if not self._graceful_mode:
                        raise ModelOnexError(
                            f"Contract validation failed at {contract_file}: {e}",
                            error_code="HANDLER_SOURCE_004",
                        ) from e
                    logger.warning(
                        "Contract validation failed, continuing in graceful mode: %s",
                        contract_file,
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
                    # Handle file size limit errors in graceful mode
                    if not self._graceful_mode:
                        raise
                    # Extract file size from error message for structured error
                    # The error message format is:
                    # "Contract file exceeds size limit: {size} bytes (max: {max} bytes)"
                    error = self._create_size_limit_error(
                        contract_file,
                        contract_file.stat().st_size,
                    )
                    logger.warning(
                        "Contract file size limit exceeded, continuing in graceful mode: %s",
                        contract_file,
                        extra={
                            "contract_file": str(contract_file),
                            "error_type": "size_limit_error",
                            "error_code": e.error_code,
                            "graceful_mode": self._graceful_mode,
                            "paths_scanned": len(self._contract_paths),
                        },
                    )
                    validation_errors.append(error)

        # Log discovery results
        self._log_discovery_results(len(descriptors), len(validation_errors))

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
            ModelOnexError: If contract file exceeds MAX_CONTRACT_SIZE (10MB).
            yaml.YAMLError: If YAML parsing fails.
            ValidationError: If contract validation fails.
        """
        # TODO(OMN-1097): Replace direct file I/O with FileRegistry abstraction
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

        # Transform to descriptor
        return ModelHandlerDescriptor(
            handler_id=contract.handler_id,
            name=contract.name,
            version=contract.version,
            handler_kind=contract.descriptor.handler_kind,
            input_model=contract.input_model,
            output_model=contract.output_model,
            description=contract.description,
            contract_path=str(contract_path),
        )

    def _create_parse_error(
        self,
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
            message=f"Failed to parse YAML: {error}",
            remediation_hint="Check YAML syntax and ensure proper indentation",
            file_path=str(contract_path),
        )

    def _create_validation_error(
        self,
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
            message=f"Contract validation failed: {error_msg} at {field_loc}",
            remediation_hint=f"Check the '{field_loc}' field in the contract",
            file_path=str(contract_path),
        )

    def _create_size_limit_error(
        self,
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
                f"Contract file exceeds size limit: {file_size} bytes "
                f"(max: {MAX_CONTRACT_SIZE} bytes)"
            ),
            remediation_hint=(
                f"Reduce contract file size to under {MAX_CONTRACT_SIZE // (1024 * 1024)}MB. "
                "Consider splitting into multiple contracts if needed."
            ),
            file_path=str(contract_path),
        )

    def _log_discovery_results(
        self,
        discovered_count: int,
        failure_count: int,
    ) -> None:
        """Log the discovery results with structured counts.

        Args:
            discovered_count: Number of successfully discovered contracts.
            failure_count: Number of validation failures.
        """
        logger.info(
            "Handler contract discovery completed: "
            "discovered_contract_count=%d, validation_failure_count=%d, "
            "paths_scanned=%d, graceful_mode=%s",
            discovered_count,
            failure_count,
            len(self._contract_paths),
            self._graceful_mode,
            extra={
                "discovered_contract_count": discovered_count,
                "validation_failure_count": failure_count,
                "paths_scanned": len(self._contract_paths),
                "graceful_mode": self._graceful_mode,
                "contract_paths": [str(p) for p in self._contract_paths],
            },
        )


__all__ = [
    "HandlerContractSource",
    "MAX_CONTRACT_SIZE",
    "ModelContractDiscoveryResult",
    "ModelHandlerDescriptor",
]
