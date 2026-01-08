# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Contract Source for Filesystem Discovery.

This module provides HandlerContractSource, which discovers handler contracts
from the filesystem by recursively scanning configured paths for
handler_contract.yaml files, parsing them, and transforming them into
ProtocolHandlerDescriptor instances.

Part of OMN-1097: HandlerContractSource + Filesystem Discovery.

The source implements ProtocolHandlerSource from omnibase_spi and supports
two operation modes:
- Strict mode (default): Raises on first error encountered
- Graceful mode: Collects errors, continues discovery, returns results with errors

See Also:
    - ProtocolHandlerSource: Protocol definition in omnibase_spi
    - ModelHandlerContract: Contract model from omnibase_core
    - ModelHandlerValidationError: Structured error model for validation failures

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
# This is tested in: tests/integration/handlers/test_handler_contract_source.py
ModelContractDiscoveryResult.model_rebuild()

logger = logging.getLogger(__name__)

# File pattern for handler contracts
HANDLER_CONTRACT_FILENAME = "handler_contract.yaml"


# =============================================================================
# HandlerContractSource Implementation
# =============================================================================


class HandlerContractSource:
    """Handler source that discovers contracts from the filesystem.

    This class implements ProtocolHandlerSource by recursively scanning
    configured paths for handler_contract.yaml files, parsing them with
    YAML and validating against ModelHandlerContract from omnibase_core.

    The source supports two operation modes:
    - Strict mode (default): Raises ModelOnexError on first error
    - Graceful mode: Collects all errors, continues discovery

    Attributes:
        source_type: Returns "CONTRACT" as the source type identifier.

    Example:
        >>> # Strict mode discovery
        >>> source = HandlerContractSource(contract_paths=[Path("./handlers")])
        >>> descriptors = await source.discover_handlers()
        >>> print(f"Found {len(descriptors)} handlers")

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
    ) -> list[ModelHandlerDescriptor] | ModelContractDiscoveryResult:
        """Discover handler contracts from configured paths.

        Recursively scans all configured paths for handler_contract.yaml files,
        parses them, validates against ModelHandlerContract, and transforms
        them into ModelHandlerDescriptor instances.

        In strict mode (default), raises on the first error encountered.
        In graceful mode, collects all errors and returns a result object.

        Returns:
            In strict mode: List of ModelHandlerDescriptor instances.
            In graceful mode: ModelContractDiscoveryResult with descriptors and errors.

        Raises:
            ModelOnexError: In strict mode, if a path doesn't exist or
                a contract fails to parse/validate.
        """
        descriptors: list[ModelHandlerDescriptor] = []
        validation_errors: list[ModelHandlerValidationError] = []
        # Track discovered files to avoid duplicates when paths overlap
        discovered_paths: set[Path] = set()

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
                logger.warning(error_msg)
                continue

            # Discover contract files
            contract_files = self._find_contract_files(base_path)

            for contract_file in contract_files:
                # Deduplicate using resolved path to handle overlapping search paths
                resolved_path = contract_file.resolve()
                if resolved_path in discovered_paths:
                    continue
                discovered_paths.add(resolved_path)

                try:
                    descriptor = self._parse_contract_file(contract_file)
                    descriptors.append(descriptor)
                except yaml.YAMLError as e:
                    error = self._create_parse_error(contract_file, e)
                    if not self._graceful_mode:
                        raise ModelOnexError(
                            f"Failed to parse YAML contract at {contract_file}: {e}",
                            error_code="HANDLER_SOURCE_003",
                        ) from e
                    validation_errors.append(error)
                except ValidationError as e:
                    error = self._create_validation_error(contract_file, e)
                    if not self._graceful_mode:
                        raise ModelOnexError(
                            f"Contract validation failed at {contract_file}: {e}",
                            error_code="HANDLER_SOURCE_004",
                        ) from e
                    validation_errors.append(error)

        # Log discovery results
        self._log_discovery_results(len(descriptors), len(validation_errors))

        if self._graceful_mode:
            return ModelContractDiscoveryResult(
                descriptors=descriptors,
                validation_errors=validation_errors,
            )

        return descriptors

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
            yaml.YAMLError: If YAML parsing fails.
            ValidationError: If contract validation fails.
        """
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
            "discovered_contract_count=%d, validation_failure_count=%d",
            discovered_count,
            failure_count,
            extra={
                "discovered_contract_count": discovered_count,
                "validation_failure_count": failure_count,
            },
        )


__all__ = [
    "HandlerContractSource",
    "ModelContractDiscoveryResult",
    "ModelHandlerDescriptor",
]
