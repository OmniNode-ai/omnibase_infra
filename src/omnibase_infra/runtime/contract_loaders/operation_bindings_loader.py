# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Loader for operation_bindings section of contract.yaml.

Validates at load time with explicit error codes. Parses expressions
into pre-compiled ModelParsedBinding instances for fast resolution.

Part of OMN-1518: Declarative operation bindings.

Contract Structure:
    The contract.yaml uses a nested structure for operation bindings::

        operation_bindings:
          version: { major: 1, minor: 0, patch: 0 }
          global_bindings:
            - parameter_name: "correlation_id"
              expression: "${envelope.correlation_id}"
          bindings:
            "db.query":
              - parameter_name: "sql"
                expression: "${payload.sql}"
              - parameter_name: "timestamp"
                expression: "${context.now_iso}"
                required: false

Usage:
    ```python
    from pathlib import Path
    from omnibase_infra.runtime.contract_loaders import (
        load_operation_bindings_subcontract,
    )

    # Load bindings from contract.yaml
    contract_path = Path("nodes/my_handler/contract.yaml")
    bindings = load_operation_bindings_subcontract(contract_path)

    # Access parsed bindings
    for operation, binding_list in bindings.bindings.items():
        for binding in binding_list:
            print(f"{operation}: {binding.parameter_name} <- {binding.original_expression}")
    ```

See Also:
    - ModelOperationBindingsSubcontract: Model for bindings configuration
    - ModelParsedBinding: Model for individual pre-parsed bindings
    - ModelOperationBinding: Raw binding entry from YAML
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.bindings import (
    EXPRESSION_PATTERN,
    MAX_EXPRESSION_LENGTH,
    MAX_PATH_SEGMENTS,
    VALID_CONTEXT_PATHS,
    VALID_SOURCES,
    ModelOperationBinding,
    ModelOperationBindingsSubcontract,
    ModelParsedBinding,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Security Constants (Loader-specific)
# =============================================================================

# Maximum allowed file size for contract.yaml files (10MB)
# Security control to prevent memory exhaustion via large YAML files
# Error code: FILE_SIZE_EXCEEDED (BINDING_LOADER_050)
MAX_CONTRACT_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB

# =============================================================================
# Error Codes
# =============================================================================

# Expression validation errors (010-019)
ERROR_CODE_EXPRESSION_MALFORMED = "BINDING_LOADER_010"
ERROR_CODE_INVALID_SOURCE = "BINDING_LOADER_011"
ERROR_CODE_PATH_TOO_DEEP = "BINDING_LOADER_012"
ERROR_CODE_EXPRESSION_TOO_LONG = "BINDING_LOADER_013"
ERROR_CODE_EMPTY_PATH_SEGMENT = "BINDING_LOADER_014"
ERROR_CODE_MISSING_PATH_SEGMENT = "BINDING_LOADER_015"
ERROR_CODE_INVALID_CONTEXT_PATH = "BINDING_LOADER_016"

# Binding validation errors (020-029)
ERROR_CODE_UNKNOWN_OPERATION = "BINDING_LOADER_020"
ERROR_CODE_DUPLICATE_PARAMETER = "BINDING_LOADER_021"

# File/contract errors (030-039)
ERROR_CODE_CONTRACT_NOT_FOUND = "BINDING_LOADER_030"
ERROR_CODE_YAML_PARSE_ERROR = "BINDING_LOADER_031"

# Security errors (050-059)
ERROR_CODE_FILE_SIZE_EXCEEDED = "BINDING_LOADER_050"


def _check_file_size(contract_path: Path, operation: str) -> None:
    """Check that contract file does not exceed maximum allowed size.

    This is a security control to prevent memory exhaustion attacks via
    oversized YAML files. Per CLAUDE.md Handler Plugin Loader security patterns,
    a 10MB file size limit is enforced.

    Args:
        contract_path: Path to the contract.yaml file.
        operation: Name of the operation for error context.

    Raises:
        ProtocolConfigurationError: If file exceeds MAX_CONTRACT_FILE_SIZE_BYTES.
            Error code: FILE_SIZE_EXCEEDED (BINDING_LOADER_050).
    """
    try:
        file_size = contract_path.stat().st_size
    except FileNotFoundError:
        # Let the caller handle FileNotFoundError with its own error message
        return

    if file_size > MAX_CONTRACT_FILE_SIZE_BYTES:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(contract_path),
        )
        logger.error(
            "Contract file exceeds maximum size: %d bytes > %d bytes at %s",
            file_size,
            MAX_CONTRACT_FILE_SIZE_BYTES,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Contract file exceeds maximum size: {file_size} bytes > "
            f"{MAX_CONTRACT_FILE_SIZE_BYTES} bytes. "
            f"Reduce the contract.yaml file size or split into multiple contracts. "
            f"Error code: FILE_SIZE_EXCEEDED ({ERROR_CODE_FILE_SIZE_EXCEEDED})",
            context=ctx,
        )


def _parse_expression(
    expression: str,
    contract_path: Path,
) -> tuple[Literal["payload", "envelope", "context"], tuple[str, ...]]:
    """Parse and validate binding expression at load time.

    Converts a binding expression (e.g., "${payload.user.id}") into its
    component parts for fast resolution at runtime. All validation happens
    here at load time, not at resolution time.

    Args:
        expression: Expression in ${source.path.to.field} format.
        contract_path: Path for error context.

    Returns:
        Tuple of (source, path_segments) where:
        - source: One of "payload", "envelope", "context"
        - path_segments: Tuple of field names to traverse

    Raises:
        ProtocolConfigurationError: With specific error code for:
        - BINDING_LOADER_010: Malformed expression syntax
        - BINDING_LOADER_011: Invalid source
        - BINDING_LOADER_012: Path too deep
        - BINDING_LOADER_013: Expression too long
        - BINDING_LOADER_014: Empty path segment
        - BINDING_LOADER_015: Missing path segment
        - BINDING_LOADER_016: Invalid context path
    """
    ctx = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.FILESYSTEM,
        operation="parse_binding_expression",
        target_name=str(contract_path),
    )

    # Check length first (before parsing)
    if len(expression) > MAX_EXPRESSION_LENGTH:
        logger.error(
            "Expression exceeds max length (%d > %d): %s in %s",
            len(expression),
            MAX_EXPRESSION_LENGTH,
            expression,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Expression exceeds max length ({len(expression)} > {MAX_EXPRESSION_LENGTH}): "
            f"{expression}. "
            f"Error code: EXPRESSION_TOO_LONG ({ERROR_CODE_EXPRESSION_TOO_LONG})",
            context=ctx,
        )

    # Check for array access (not supported)
    if "[" in expression or "]" in expression:
        logger.error(
            "Array access not allowed in expressions: %s in %s",
            expression,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Array access not allowed in expressions: {expression}. "
            f"Use path-based access only (e.g., ${{payload.items}} not ${{payload.items[0]}}). "
            f"Error code: EXPRESSION_MALFORMED ({ERROR_CODE_EXPRESSION_MALFORMED})",
            context=ctx,
        )

    # Parse expression using regex
    match = EXPRESSION_PATTERN.match(expression)
    if not match:
        logger.error(
            "Invalid expression syntax: %s in %s. Expected ${{source.path.to.field}}",
            expression,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Invalid expression syntax: {expression}. "
            f"Expected format: ${{source.path.to.field}}. "
            f"Error code: EXPRESSION_MALFORMED ({ERROR_CODE_EXPRESSION_MALFORMED})",
            context=ctx,
        )

    source = match.group(1)
    path_str = match.group(2)

    # Validate source is one of the allowed values
    if source not in VALID_SOURCES:
        logger.error(
            "Invalid source '%s' in expression %s at %s. Must be one of: %s",
            source,
            expression,
            contract_path,
            sorted(VALID_SOURCES),
        )
        raise ProtocolConfigurationError(
            f"Invalid source '{source}' in expression: {expression}. "
            f"Must be one of: {sorted(VALID_SOURCES)}. "
            f"Error code: INVALID_SOURCE ({ERROR_CODE_INVALID_SOURCE})",
            context=ctx,
        )

    # Parse path segments
    path_segments = tuple(path_str.split("."))

    # Check for empty segments (e.g., "payload..id" or "payload.")
    if any(segment == "" for segment in path_segments):
        logger.error(
            "Empty path segment in expression: %s at %s",
            expression,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Empty path segment in expression: {expression}. "
            f"Path segments cannot be empty. "
            f"Error code: EMPTY_PATH_SEGMENT ({ERROR_CODE_EMPTY_PATH_SEGMENT})",
            context=ctx,
        )

    # Check minimum path (must have at least one segment)
    if len(path_segments) == 0:
        logger.error(
            "Expression must have at least one path segment: %s at %s",
            expression,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Expression must have at least one path segment: {expression}. "
            f"Error code: MISSING_PATH_SEGMENT ({ERROR_CODE_MISSING_PATH_SEGMENT})",
            context=ctx,
        )

    # Check max segments (prevent deep nesting attacks)
    if len(path_segments) > MAX_PATH_SEGMENTS:
        logger.error(
            "Path exceeds max segments (%d > %d) in expression: %s at %s",
            len(path_segments),
            MAX_PATH_SEGMENTS,
            expression,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Path exceeds max segments ({len(path_segments)} > {MAX_PATH_SEGMENTS}): "
            f"{expression}. "
            f"Error code: PATH_TOO_DEEP ({ERROR_CODE_PATH_TOO_DEEP})",
            context=ctx,
        )

    # Validate context paths (only certain values are allowed)
    if source == "context" and path_segments[0] not in VALID_CONTEXT_PATHS:
        logger.error(
            "Invalid context path '%s' in expression %s at %s. Must be one of: %s",
            path_segments[0],
            expression,
            contract_path,
            sorted(VALID_CONTEXT_PATHS),
        )
        raise ProtocolConfigurationError(
            f"Invalid context path '{path_segments[0]}' in expression: {expression}. "
            f"Must be one of: {sorted(VALID_CONTEXT_PATHS)}. "
            f"Error code: INVALID_CONTEXT_PATH ({ERROR_CODE_INVALID_CONTEXT_PATH})",
            context=ctx,
        )

    # Type assertion for mypy - source is guaranteed to be one of the valid values
    return source, path_segments  # type: ignore[return-value]


def _parse_binding_entry(
    raw_binding: dict[str, object],
    contract_path: Path,
) -> ModelParsedBinding:
    """Parse a raw binding dict into ModelParsedBinding.

    First validates the raw YAML structure using ModelOperationBinding,
    then parses the expression into pre-compiled components.

    Args:
        raw_binding: Raw binding dict from YAML.
        contract_path: Path for error context.

    Returns:
        ModelParsedBinding with pre-parsed expression components.

    Raises:
        ProtocolConfigurationError: If binding or expression is invalid.
        ValidationError: If raw binding doesn't match ModelOperationBinding schema.
    """
    # First validate as ModelOperationBinding (raw YAML structure)
    # This validates required fields and types
    operation_binding = ModelOperationBinding(**raw_binding)

    # Parse the expression into components
    source, path_segments = _parse_expression(
        operation_binding.expression,
        contract_path,
    )

    return ModelParsedBinding(
        parameter_name=operation_binding.parameter_name,
        source=source,
        path_segments=path_segments,
        required=operation_binding.required,
        default=operation_binding.default,
        original_expression=operation_binding.expression,
    )


def _check_duplicate_parameters(
    bindings: list[ModelParsedBinding],
    scope: str,
    contract_path: Path,
) -> None:
    """Check for duplicate parameter names within a binding list.

    Args:
        bindings: List of parsed bindings to check.
        scope: Description of scope for error message (e.g., "global_bindings").
        contract_path: Path for error context.

    Raises:
        ProtocolConfigurationError: If duplicate parameter name found.
            Error code: DUPLICATE_PARAMETER (BINDING_LOADER_021).
    """
    seen: set[str] = set()
    for binding in bindings:
        if binding.parameter_name in seen:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="validate_bindings",
                target_name=str(contract_path),
            )
            logger.error(
                "Duplicate parameter '%s' in %s at %s",
                binding.parameter_name,
                scope,
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Duplicate parameter '{binding.parameter_name}' in {scope}. "
                f"Each parameter name must be unique within its scope. "
                f"Error code: DUPLICATE_PARAMETER ({ERROR_CODE_DUPLICATE_PARAMETER})",
                context=ctx,
            )
        seen.add(binding.parameter_name)


def load_operation_bindings_subcontract(
    contract_path: Path,
    io_operations: list[str] | None = None,
) -> ModelOperationBindingsSubcontract:
    """Load, parse, and validate operation_bindings from contract.yaml.

    Loads the operation_bindings section from a contract.yaml file
    and converts it to ModelOperationBindingsSubcontract format with
    pre-parsed expressions. All validation happens at load time.

    Validation at load time:
    - File size limit (10MB) - security control
    - YAML safe_load - security control
    - Expression syntax validation
    - Source validation (payload/envelope/context)
    - Context path validation (now_iso/dispatcher_id/correlation_id)
    - Duplicate parameter detection per scope
    - io_operations reference validation (if provided)

    Args:
        contract_path: Path to contract.yaml file.
        io_operations: Optional list of valid operation names. If provided,
            validates that all operation names in bindings exist in this list.

    Returns:
        ModelOperationBindingsSubcontract with pre-parsed bindings.
        Returns empty subcontract if operation_bindings section is missing.

    Raises:
        ProtocolConfigurationError: With specific error code for various failures:
        - BINDING_LOADER_030: Contract file not found
        - BINDING_LOADER_031: YAML parse error
        - BINDING_LOADER_050: File size exceeded
        - BINDING_LOADER_010-016: Expression validation errors
        - BINDING_LOADER_020: Unknown operation (not in io_operations)
        - BINDING_LOADER_021: Duplicate parameter name

    Example:
        ```python
        from pathlib import Path
        from omnibase_infra.runtime.contract_loaders import (
            load_operation_bindings_subcontract,
        )

        contract_path = Path(__file__).parent / "contract.yaml"
        bindings = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["db.query", "db.execute"],  # Optional validation
        )

        # Access parsed bindings
        for op_name, binding_list in bindings.bindings.items():
            for binding in binding_list:
                print(f"{op_name}: {binding.parameter_name}")
        ```
    """
    operation = "load_operation_bindings"
    ctx = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.FILESYSTEM,
        operation=operation,
        target_name=str(contract_path),
    )

    # Check file exists
    if not contract_path.exists():
        logger.error(
            "Contract file not found: %s - cannot load operation bindings",
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Contract file not found: {contract_path}. "
            f"Ensure the contract.yaml exists in the handler directory. "
            f"Error code: CONTRACT_NOT_FOUND ({ERROR_CODE_CONTRACT_NOT_FOUND})",
            context=ctx,
        )

    # Check file size (security control - MUST be before yaml.safe_load)
    _check_file_size(contract_path, operation)

    # Load YAML safely
    try:
        with contract_path.open("r", encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # Sanitize error message - don't include raw YAML error which may contain file contents
        error_type = type(e).__name__
        logger.exception(
            "Invalid YAML syntax in contract.yaml at %s: %s",
            contract_path,
            error_type,
        )
        raise ProtocolConfigurationError(
            f"Invalid YAML syntax in contract.yaml at {contract_path}: {error_type}. "
            f"Verify the YAML syntax is correct. "
            f"Error code: YAML_PARSE_ERROR ({ERROR_CODE_YAML_PARSE_ERROR})",
            context=ctx,
        ) from e

    if contract_data is None:
        contract_data = {}

    # Get operation_bindings section (optional - return empty if missing)
    bindings_section = contract_data.get("operation_bindings", {})
    if not bindings_section:
        logger.debug(
            "No operation_bindings section in contract.yaml at %s - returning empty subcontract",
            contract_path,
        )
        return ModelOperationBindingsSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            bindings={},
            global_bindings=None,
        )

    # Parse version
    version_data = bindings_section.get("version", {"major": 1, "minor": 0, "patch": 0})
    if isinstance(version_data, dict):
        version = ModelSemVer(**version_data)
    else:
        version = ModelSemVer(major=1, minor=0, patch=0)

    # Parse global_bindings (optional)
    global_bindings: list[ModelParsedBinding] | None = None
    raw_global = bindings_section.get("global_bindings", [])
    if raw_global:
        global_bindings = [_parse_binding_entry(b, contract_path) for b in raw_global]
        _check_duplicate_parameters(global_bindings, "global_bindings", contract_path)
        logger.debug(
            "Loaded %d global bindings from contract.yaml at %s",
            len(global_bindings),
            contract_path,
        )

    # Parse operation-specific bindings
    parsed_bindings: dict[str, list[ModelParsedBinding]] = {}
    raw_bindings = bindings_section.get("bindings", {})

    for operation_name, operation_binding_list in raw_bindings.items():
        # Validate operation exists in io_operations (if provided)
        if io_operations is not None and operation_name not in io_operations:
            logger.error(
                "Unknown operation '%s' in bindings - not in io_operations at %s",
                operation_name,
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Unknown operation '{operation_name}' in bindings section. "
                f"Not found in io_operations: {sorted(io_operations)}. "
                f"Error code: UNKNOWN_OPERATION ({ERROR_CODE_UNKNOWN_OPERATION})",
                context=ctx,
            )

        # Parse all bindings for this operation
        parsed_list = [
            _parse_binding_entry(b, contract_path) for b in operation_binding_list
        ]

        # Check for duplicates within this operation's bindings
        _check_duplicate_parameters(
            parsed_list, f"operation '{operation_name}'", contract_path
        )

        parsed_bindings[operation_name] = parsed_list

    logger.debug(
        "Loaded %d operation binding groups from contract.yaml at %s",
        len(parsed_bindings),
        contract_path,
    )

    return ModelOperationBindingsSubcontract(
        version=version,
        bindings=parsed_bindings,
        global_bindings=global_bindings,
    )


__all__ = [
    # Loader-specific constants
    "MAX_CONTRACT_FILE_SIZE_BYTES",
    # Error codes
    "ERROR_CODE_CONTRACT_NOT_FOUND",
    "ERROR_CODE_DUPLICATE_PARAMETER",
    "ERROR_CODE_EMPTY_PATH_SEGMENT",
    "ERROR_CODE_EXPRESSION_MALFORMED",
    "ERROR_CODE_EXPRESSION_TOO_LONG",
    "ERROR_CODE_FILE_SIZE_EXCEEDED",
    "ERROR_CODE_INVALID_CONTEXT_PATH",
    "ERROR_CODE_INVALID_SOURCE",
    "ERROR_CODE_MISSING_PATH_SEGMENT",
    "ERROR_CODE_PATH_TOO_DEEP",
    "ERROR_CODE_UNKNOWN_OPERATION",
    "ERROR_CODE_YAML_PARSE_ERROR",
    # Main function
    "load_operation_bindings_subcontract",
]
